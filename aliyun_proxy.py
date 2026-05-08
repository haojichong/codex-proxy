import getpass
import json
import os
import sys
import uuid

import requests
import urllib3
from flask import Flask, request, Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress SSL retry warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _create_session():
    """Create a requests session with aggressive retry for SSL stability"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[502, 503, 504, 429],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=1,
        pool_maxsize=1,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Disable keep-alive to avoid stale connections causing SSL EOF
    session.headers.update({"Connection": "close"})
    return session


def _get_fresh_connection():
    """Get a fresh session for each streaming request to avoid SSL issues"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[502, 503, 504, 429],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Connection": "close"})
    return session


def _load_config():
    cfg_path = os.path.join(BASE_DIR, "config.json")
    if not os.path.exists(cfg_path):
        print("ERROR: config.json not found")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("aliyun", {})


def _ensure_api_key(cfg):
    key = cfg.get("api_key", "").strip()
    if key:
        return key, "config.json"

    print("=" * 60)
    print("  Aliyun API Key not configured")
    print("=" * 60)
    print()
    print("  Get API Key from Aliyun Bailian platform")
    print()

    try:
        key = getpass.getpass("  Enter your Aliyun API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if not key:
        print("\n  ERROR: No API Key provided. Exiting.")
        sys.exit(1)

    cfg["api_key"] = key
    cfg_path = os.path.join(BASE_DIR, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        all_cfg = json.load(f)
    all_cfg["aliyun"]["api_key"] = key
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(all_cfg, f, indent=2, ensure_ascii=False)
    print(f"\n  API Key saved to config.json\n")
    return key, "config.json (saved)"


_CFG = _load_config()

DEBUG_LOG = os.path.join(BASE_DIR, "debug_aliyun.log")

app = Flask(__name__)

# ===================== Configuration =====================
ALIYUN_API_TOKEN = _CFG.get("api_key", "").strip()
ALIYUN_MODEL = _CFG.get("model", "glm-5").strip()
ALIYUN_BASE_URL = _CFG.get("base_url", "https://coding.dashscope.aliyuncs.com/apps/anthropic").strip()
ALIYUN_API_URL = f"{ALIYUN_BASE_URL}/v1/messages"
ALIYUN_DEBUG = _CFG.get("debug", False)
AUTH_TYPE = _CFG.get("auth_type", "bearer")
PORT = _CFG.get("port", 5002)
API_TIMEOUT_MS = 3000000


# =================================================


def _clean_schema(obj):
    """递归清除 JSON Schema 中不支持的字段
    
    移除 Anthropic API 不支持的字段：additionalProperties, strict
    """
    if not isinstance(obj, dict):
        return obj
    cleaned = {}
    for k, v in obj.items():
        if k in ("additionalProperties", "strict"):
            continue
        if isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


def _convert_tools_to_anthropic(tools: list) -> list:
    """将 Responses API 格式的工具定义转换为 Anthropic 格式
    
    Responses API 工具格式: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    Anthropic 工具格式: {"name": "...", "description": "...", "input_schema": {...}}
    """
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        # Responses API 格式：name/description/parameters 在顶层
        # 同时兼容 OpenAI Chat Completions 格式：在 function 子对象中
        name = tool.get("name", "") or tool.get("function", {}).get("name", "")
        description = tool.get("description", "") or tool.get("function", {}).get("description", "")
        parameters = tool.get("parameters") or tool.get("function", {}).get("parameters")
        
        anthropic_tool = {
            "name": name,
            "description": description,
        }
        if parameters:
            anthropic_tool["input_schema"] = _clean_schema(parameters)
        result.append(anthropic_tool)
    return result


def _convert_tool_choice_anthropic(tc):
    """将 Responses API tool_choice 转换为 Anthropic 格式
    
    Responses API 格式: {"type": "function", "name": "..."}
    Anthropic 格式: {"type": "tool", "name": "..."}
    """
    if tc is None:
        return {"type": "auto"}
    if isinstance(tc, str):
        if tc == "auto":
            return {"type": "auto"}
        if tc == "none":
            return {"type": "any"}
        if tc == "required":
            return {"type": "any"}
    if isinstance(tc, dict) and tc.get("type") == "function":
        # Responses API 格式：name 在顶层
        # 同时兼容 OpenAI Chat Completions 格式：name 在 function 子对象中
        func_name = tc.get("name", "") or tc.get("function", {}).get("name", "")
        if func_name:
            return {"type": "tool", "name": func_name}
    return {"type": "auto"}


def extract_messages_for_anthropic(data: dict):
    """从 Responses API 请求中提取消息，转换为 Anthropic Messages API 格式"""
    ROLE_MAP = {"developer": "user"}
    raw_tools = data.get("tools", [])
    tools = _convert_tools_to_anthropic(raw_tools)
    tool_choice = _convert_tool_choice_anthropic(data.get("tool_choice"))

    system_prompt = ""
    messages = []

    if "instructions" in data and data["instructions"]:
        system_prompt = data["instructions"]

    if "input" not in data:
        if "messages" in data:
            for msg in data["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    if system_prompt:
                        system_prompt += "\n" + (content if isinstance(content, str) else str(content))
                    else:
                        system_prompt = content if isinstance(content, str) else str(content)
                    continue
                role = ROLE_MAP.get(role, role)
                if role not in ("user", "assistant"):
                    role = "user"
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    anthropic_content = []
                    for c in content:
                        if isinstance(c, dict):
                            c_type = c.get("type", "")
                            if c_type == "text":
                                anthropic_content.append({"type": "text", "text": c.get("text", "")})
                            elif c_type == "image_url":
                                img_url = c.get("image_url", {})
                                url = img_url.get("url", "")
                                if url.startswith("data:"):
                                    mime_end = url.find(";base64,")
                                    if mime_end > 0:
                                        media_type = url[5:mime_end]
                                        base64_data = url[mime_end + 8:]
                                        anthropic_content.append({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": base64_data
                                            }
                                        })
                    if anthropic_content:
                        messages.append({"role": role, "content": anthropic_content})
        return system_prompt, messages, tools, tool_choice

    inp = data["input"]
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
        return system_prompt, messages, tools, tool_choice

    if not isinstance(inp, list):
        return system_prompt, messages, tools, tool_choice

    pending_tool_calls = []

    def _flush_tool_calls():
        """将累积的 tool_calls 合并为一个 assistant 消息"""
        nonlocal pending_tool_calls
        if pending_tool_calls:
            content_blocks = []
            for tc in pending_tool_calls:
                # 安全解析 JSON，失败时使用空对象
                try:
                    input_data = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    input_data = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": input_data,
                })
            messages.append({"role": "assistant", "content": content_blocks})
            pending_tool_calls = []

    for item in inp:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "message":
            _flush_tool_calls()
            role = item.get("role", "user")
            role = ROLE_MAP.get(role, role)
            if role not in ("user", "assistant"):
                role = "user"
            content = item.get("content", "")

            if isinstance(content, list):
                text_parts = []
                tool_uses = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    c_type = c.get("type")
                    if c_type in ("text", "input_text", "output_text"):
                        t = c.get("text", "")
                        if t.strip():
                            text_parts.append(t)
                    elif c_type == "tool_call":
                        # 安全解析 JSON
                        try:
                            tool_input = json.loads(c.get("arguments", "{}")) if c.get("arguments") else {}
                        except json.JSONDecodeError:
                            tool_input = {}
                        tool_uses.append({
                            "type": "tool_use",
                            "id": c.get("id", ""),
                            "name": c.get("name", ""),
                            "input": tool_input,
                        })

                content_blocks = []
                if text_parts:
                    content_blocks.append({"type": "text", "text": "\n".join(text_parts)})
                content_blocks.extend(tool_uses)
                if content_blocks:
                    messages.append({"role": role, "content": content_blocks})

            elif isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})

        elif item_type == "function_call":
            pending_tool_calls.append({
                "id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            })

        elif item_type == "function_call_output":
            _flush_tool_calls()
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                }]
            })

    _flush_tool_calls()

    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "请继续"})

    return system_prompt, messages, tools, tool_choice


# ---- CORS ----
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp


# ---- 路由处理 ----
def _make_response():
    """处理 /responses 系列请求的核心逻辑"""
    if request.method == "OPTIONS":
        return Response()

    req_data = request.get_json(silent=True) or {}
    system_prompt, messages, tools, tool_choice = extract_messages_for_anthropic(req_data)
    effective_model = ALIYUN_MODEL
    response_id = f"resp_{uuid.uuid4().hex[:12]}"

    if ALIYUN_DEBUG:
        debug_path = request.path
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n--- [{__import__('datetime').datetime.now()}] PATH={debug_path} ---\n")
            f.write(f"Request body:\n{json.dumps(req_data, indent=2, ensure_ascii=False)}\n")
            f.write(f"System: {system_prompt[:500]}...\n" if len(system_prompt) > 500 else f"System: {system_prompt}\n")
            f.write(f"Messages:\n{json.dumps(messages, indent=2, ensure_ascii=False)}\n")
            if tools:
                f.write(f"Tools count: {len(tools)}\n")
                f.write(f"Tool choice: {tool_choice}\n")

    def generate():
        if not messages:
            yield "event: response.completed\n"
            yield (
                    "data: "
                    + json.dumps({
                "type": "response.completed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "completed", "model": effective_model,
                    "output": [], "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            }, ensure_ascii=False)
                    + "\n\n"
            )
            return

        yield "event: response.created\n"
        yield (
                "data: "
                + json.dumps({
            "type": "response.created",
            "response": {
                "id": response_id, "object": "response",
                "status": "in_progress", "model": effective_model,
                "output": [], "usage": None,
            },
        }, ensure_ascii=False)
                + "\n\n"
        )

        yield "event: response.in_progress\n"
        yield (
                "data: "
                + json.dumps({
            "type": "response.in_progress",
            "response": {
                "id": response_id, "object": "response",
                "status": "in_progress", "model": effective_model,
                "output": [], "usage": None,
            },
        }, ensure_ascii=False)
                + "\n\n"
        )

        # 阿里云使用 Authorization Bearer token 认证
        headers = {
            "Authorization": f"Bearer {ALIYUN_API_TOKEN}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        payload = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        text_item_id = f"item_{uuid.uuid4().hex[:12]}"
        full_text = ""
        full_thinking = ""
        has_text = False
        text_started = False

        tool_calls_acc = {}
        tool_call_index = 0

        input_tokens = 0
        output_tokens = 0
        seq = 0

        upstream = None
        try:
            upstream = _get_fresh_connection().post(
                ALIYUN_API_URL, headers=headers, json=payload,
                stream=True, timeout=(30, API_TIMEOUT_MS / 1000),
            )

            if ALIYUN_DEBUG:
                with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                    f.write(f"Upstream status: {upstream.status_code}\n")
                    f.write(f"Upstream headers: {dict(upstream.headers)}\n")
                    if upstream.status_code != 200:
                        f.write(f"Error response: {upstream.text[:2000]}\n")

            upstream.raise_for_status()

            for line in upstream.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                # Aliyun sends "data:{...}" (no space), Anthropic sends "data: {...}"
                if line.startswith("data: "):
                    raw = line[6:].strip()
                elif line.startswith("data:"):
                    raw = line[5:].strip()
                else:
                    continue
                if not raw:
                    continue

                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    if ALIYUN_DEBUG:
                        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                            f.write(f"JSON decode error for: {raw[:200]}\n")
                    continue

                event_type = event.get("type", "")

                if event_type == "message_start":
                    usage = event.get("message", {}).get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)

                elif event_type == "content_block_start":
                    content_block = event.get("content_block", {})
                    block_type = content_block.get("type", "")

                    if block_type == "text":
                        text_started = True
                        has_text = True
                        yield "event: response.output_item.added\n"
                        yield (
                                "data: "
                                + json.dumps({
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": text_item_id, "type": "message",
                                "status": "in_progress", "role": "assistant",
                                "content": [],
                            },
                        }, ensure_ascii=False)
                                + "\n\n"
                        )
                        yield "event: response.content_part.added\n"
                        yield (
                                "data: "
                                + json.dumps({
                            "type": "response.content_part.added",
                            "item_id": text_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "text", "text": ""},
                        }, ensure_ascii=False)
                                + "\n\n"
                        )

                    elif block_type == "tool_use":
                        tool_id = content_block.get("id", "")
                        tool_name = content_block.get("name", "")
                        tool_calls_acc[tool_id] = {
                            "name": tool_name,
                            "input": "",
                            "item_id": f"item_{uuid.uuid4().hex[:12]}",
                            "started": True,
                            "index": tool_call_index,
                        }
                        tool_call_index += 1
                        out_idx = (1 if has_text else 0) + sorted(
                            [t["index"] for t in tool_calls_acc.values()]
                        ).index(tool_calls_acc[tool_id]["index"])

                        yield "event: response.output_item.added\n"
                        yield (
                                "data: "
                                + json.dumps({
                            "type": "response.output_item.added",
                            "output_index": out_idx,
                            "item": {
                                "id": tool_calls_acc[tool_id]["item_id"],
                                "type": "function_call",
                                "status": "in_progress",
                                "call_id": tool_id,
                                "name": tool_name,
                                "arguments": "",
                            },
                        }, ensure_ascii=False)
                                + "\n\n"
                        )

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    block_index = event.get("index", 0)

                    if delta.get("type") == "thinking_delta":
                        thinking_text = delta.get("thinking", "")
                        if thinking_text:
                            full_thinking += thinking_text

                    elif delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            full_text += text
                            seq += 1
                            yield "event: response.output_text.delta\n"
                            yield (
                                    "data: "
                                    + json.dumps({
                                "type": "response.output_text.delta",
                                "delta": text,
                                "item_id": text_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "sequence_number": seq,
                            }, ensure_ascii=False)
                                    + "\n\n"
                            )

                    elif delta.get("type") == "input_json_delta":
                        partial_json = delta.get("partial_json", "")
                        if partial_json and block_index is not None:
                            for tid, acc in tool_calls_acc.items():
                                if acc["index"] == block_index:
                                    acc["input"] += partial_json
                                    out_idx = (1 if has_text else 0) + sorted(
                                        [t["index"] for t in tool_calls_acc.values()]
                                    ).index(acc["index"])
                                    yield "event: response.function_call_arguments.delta\n"
                                    yield (
                                            "data: "
                                            + json.dumps({
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": acc["item_id"],
                                        "output_index": out_idx,
                                        "delta": partial_json,
                                    }, ensure_ascii=False)
                                            + "\n\n"
                                    )
                                    break

                elif event_type == "message_delta":
                    usage = event.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                elif event_type == "content_block_stop":
                    pass

                elif event_type == "message_stop":
                    pass

            # ===== 流结束后发出完成事件 =====

            if has_text:
                yield "event: response.output_text.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.output_text.done",
                    "text": full_text, "item_id": text_item_id,
                    "output_index": 0, "content_index": 0,
                }, ensure_ascii=False)
                        + "\n\n"
                )
                yield "event: response.content_part.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.content_part.done",
                    "item_id": text_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "text", "text": full_text},
                }, ensure_ascii=False)
                        + "\n\n"
                )
                yield "event: response.output_item.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": text_item_id, "type": "message",
                        "status": "completed", "role": "assistant",
                        "content": [{"type": "text", "text": full_text}],
                        **({"reasoning_content": full_thinking} if full_thinking else {}),
                    },
                }, ensure_ascii=False)
                        + "\n\n"
                )

            output_items = []
            if has_text:
                output_items.append({
                    "id": text_item_id, "type": "message",
                    "status": "completed", "role": "assistant",
                    "content": [{"type": "text", "text": full_text}],
                    **({"reasoning_content": full_thinking} if full_thinking else {}),
                })

            for tid, acc in sorted(tool_calls_acc.items(), key=lambda x: x[1]["index"]):
                out_idx = (1 if has_text else 0) + acc["index"]

                yield "event: response.function_call_arguments.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.function_call_arguments.done",
                    "item_id": acc["item_id"],
                    "output_index": out_idx,
                    "arguments": acc["input"],
                }, ensure_ascii=False)
                        + "\n\n"
                )

                yield "event: response.output_item.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": out_idx,
                    "item": {
                        "id": acc["item_id"],
                        "type": "function_call",
                        "status": "completed",
                        "call_id": tid,
                        "name": acc["name"],
                        "arguments": acc["input"],
                    },
                }, ensure_ascii=False)
                        + "\n\n"
                )

                output_items.append({
                    "id": acc["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tid,
                    "name": acc["name"],
                    "arguments": acc["input"],
                })

            yield "event: response.completed\n"
            yield (
                    "data: "
                    + json.dumps({
                "type": "response.completed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "completed", "model": effective_model,
                    "output": output_items,
                    "usage": {
                        "input_tokens": input_tokens or max(1, len(json.dumps(messages)) // 4),
                        "output_tokens": output_tokens or max(1, len(full_text) // 4),
                        "total_tokens": (input_tokens + output_tokens) or max(1, len(json.dumps(messages)) // 4 + len(
                            full_text) // 4),
                    },
                },
            }, ensure_ascii=False)
                    + "\n\n"
            )

        except requests.exceptions.HTTPError as e:
            body = ""
            try:
                if upstream is not None:
                    body = upstream.text[:2000]
            except Exception:
                body = "(unable to read error body)"
            err_msg = f"Aliyun API {e.response.status_code}: {body}"
            if ALIYUN_DEBUG:
                with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                    f.write(f"ERROR: {err_msg}\n")
                    f.write(f"Payload sent:\n")
                    f.write(json.dumps(payload, indent=2, ensure_ascii=False)[:5000] + "\n")
            yield "event: response.failed\n"
            yield "data: " + json.dumps({
                "type": "response.failed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "failed", "model": effective_model,
                    "error": {"message": err_msg, "type": "upstream_error"},
                    "output": [], "usage": None,
                },
            }, ensure_ascii=False) + "\n\n"

        except requests.exceptions.RequestException as e:
            yield "event: response.failed\n"
            yield "data: " + json.dumps({
                "type": "response.failed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "failed", "model": effective_model,
                    "error": {"message": str(e), "type": "upstream_error"},
                    "output": [], "usage": None,
                },
            }, ensure_ascii=False) + "\n\n"

        finally:
            if upstream is not None:
                try:
                    upstream.close()
                except Exception:
                    pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- 注册路由 ----
app.add_url_rule("/responses", "responses", _make_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/v1/responses", "v1_responses", _make_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/v1/chat/completions", "v1_chat", _make_response, methods=["POST", "OPTIONS"])

if __name__ == "__main__":
    key, source = _ensure_api_key(_CFG)
    if not key:
        sys.exit(1)
    globals()["ALIYUN_API_TOKEN"] = key

    from waitress import serve

    print("aliyun_proxy starting ...")
    print(f"   Endpoint: http://127.0.0.1:{PORT}")
    print(f"   Model:    {ALIYUN_MODEL}")
    print(f"   API URL:  {ALIYUN_API_URL}")
    print(f"   Key:      {source}")
    print(f"   Debug:    {'ON' if ALIYUN_DEBUG else 'OFF'}")
    print(f"   Routes:   /responses, /v1/responses, /v1/chat/completions")
    serve(app, host="127.0.0.1", port=PORT, threads=100, channel_timeout=300, cleanup_interval=30)
