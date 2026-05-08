import getpass
import json
import os
import sys
import uuid

import requests
from flask import Flask, request, Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_HTTP_SESSION = _create_session()


def _load_config():
    """从 config.json 加载 deepseek 配置"""
    cfg_path = os.path.join(BASE_DIR, "config.json")
    if not os.path.exists(cfg_path):
        print("ERROR: config.json not found")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("deepseek", {})


def _ensure_api_key(cfg):
    """确保 api_key 已设置，未设置则交互输入并保存"""
    key = cfg.get("api_key", "").strip()
    if key:
        return key, "config.json"

    print("=" * 60)
    print("  DeepSeek API Key not configured")
    print("=" * 60)
    print()
    print("  Get API Key from: https://platform.deepseek.com/api_keys")
    print()

    try:
        key = getpass.getpass("  Enter your DeepSeek API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if not key:
        print("\n  ERROR: No API Key provided. Exiting.")
        sys.exit(1)

    # Save to config.json
    cfg["api_key"] = key
    cfg_path = os.path.join(BASE_DIR, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        all_cfg = json.load(f)
    all_cfg["deepseek"]["api_key"] = key
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(all_cfg, f, indent=2, ensure_ascii=False)
    print(f"\n  API Key saved to config.json\n")
    return key, "config.json (saved)"


_CFG = _load_config()

DEBUG_LOG = os.path.join(BASE_DIR, "debug_deepseek.log")

app = Flask(__name__)

# ===================== Configuration =====================
DEEPSEEK_API_KEY = _CFG.get("api_key", "").strip()
DEEPSEEK_MODEL = _CFG.get("model", "deepseek-v4-pro").strip()
DEEPSEEK_URL = _CFG.get("api_url", "https://api.deepseek.com/v1/chat/completions").strip()
DEEPSEEK_DEBUG = _CFG.get("debug", False)
PORT = _CFG.get("port", 5000)


# =================================================


def _clean_schema(obj):
    """递归清除 JSON Schema 中 DeepSeek 不支持的字段"""
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


def _convert_tools(tools: list) -> list:
    """将工具定义从 Responses API 格式转换为 Chat Completions API 格式"""
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        func = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
        }
        if "parameters" in tool:
            func["parameters"] = _clean_schema(tool["parameters"])
        result.append({"type": "function", "function": func})
    return result


def _convert_tool_choice(tc):
    """将 tool_choice 从 Responses API 格式转换为 Chat Completions 格式"""
    if tc is None:
        return "auto"
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return "auto"


def extract_messages(data: dict):
    """
    从 Responses API 请求中提取 messages 列表、tools 列表和 tool_choice。
    返回: (messages, tools, tool_choice)
    """
    ROLE_MAP = {"developer": "system"}
    raw_tools = data.get("tools", [])
    tools = _convert_tools(raw_tools)
    tool_choice = _convert_tool_choice(data.get("tool_choice"))

    if "input" not in data:
        if "messages" in data:
            return data["messages"], tools, tool_choice
        return [], tools, tool_choice

    inp = data["input"]
    if isinstance(inp, str):
        messages = []
        if "instructions" in data and data["instructions"]:
            messages.append({"role": "system", "content": data["instructions"]})
        messages.append({"role": "user", "content": inp})
        return messages, tools, tool_choice

    if not isinstance(inp, list):
        return [], tools, tool_choice

    messages = []
    if "instructions" in data and data["instructions"]:
        messages.append({"role": "system", "content": data["instructions"]})

    pending_tool_calls = []  # 收集连续的 function_call
    pending_reasoning = ""  # function_call 携带的 reasoning_content

    def _flush_tool_calls():
        """将累积的 function_call 合并为一个 assistant 消息"""
        nonlocal pending_tool_calls, pending_reasoning
        if pending_tool_calls:
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": pending_tool_calls,
            }
            if pending_reasoning:
                msg["reasoning_content"] = pending_reasoning
            messages.append(msg)
            pending_tool_calls = []
            pending_reasoning = ""

    for item in inp:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "message":
            _flush_tool_calls()  # 遇到非 function_call 项，先刷新累积的 tool calls
            role = item.get("role", "user")
            role = ROLE_MAP.get(role, role)
            content = item.get("content", "")
            if isinstance(content, list):
                texts = []
                tool_calls = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    c_type = c.get("type")
                    if c_type in ("text", "input_text", "output_text"):
                        t = c.get("text", "")
                        if t.strip():
                            texts.append(t)
                    elif c_type == "tool_call":
                        tool_calls.append({
                            "id": c.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": c.get("name", ""),
                                "arguments": c.get("arguments", ""),
                            }
                        })
                text_content = "\n".join(texts)
                if tool_calls:
                    msg = {"role": role, "content": text_content or ""}
                    msg["tool_calls"] = tool_calls
                    if item.get("reasoning_content"):
                        msg["reasoning_content"] = item["reasoning_content"]
                    messages.append(msg)
                elif text_content:
                    msg = {"role": role, "content": text_content}
                    if item.get("reasoning_content"):
                        msg["reasoning_content"] = item["reasoning_content"]
                    messages.append(msg)
            elif isinstance(content, str) and content.strip():
                msg = {"role": role, "content": content.strip()}
                if item.get("reasoning_content"):
                    msg["reasoning_content"] = item["reasoning_content"]
                messages.append(msg)

        elif item_type == "function_call":
            # 累积连续的 function_call，稍后合并为一个 assistant 消息
            pending_tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                }
            })
            # 保留 reasoning_content（DeepSeek 思考模式必须回传）
            if item.get("reasoning_content") and not pending_reasoning:
                pending_reasoning = item["reasoning_content"]

        elif item_type == "function_call_output":
            _flush_tool_calls()  # function_call 序列结束，刷新
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })

    _flush_tool_calls()  # 处理末尾的 function_call

    # ---- 重排消息：确保 tool 消息紧跟对应的 assistant 消息 ----
    # DeepSeek 要求：带 tool_calls 的 assistant 消息后必须紧跟所有对应的 tool 消息，
    # 中间不能插入 system/user 消息。Codex 有时会在中间注入系统消息（如审批通知）。
    reordered = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # 收集此 assistant 需要的 tool_call_id
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            tool_msgs = []
            non_tool_msgs = []
            j = i + 1
            while j < len(messages) and expected_ids:
                nxt = messages[j]
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") in expected_ids:
                    expected_ids.remove(nxt["tool_call_id"])
                    tool_msgs.append(nxt)
                elif nxt.get("role") in ("system", "developer"):
                    non_tool_msgs.append(nxt)  # system 消息移到 assistant 之前
                else:
                    break  # user/assistant 消息——停止搜索，保持原顺序
                j += 1
            # 先放非 tool 消息（system/user），再放 assistant，最后放 tool 消息
            reordered.extend(non_tool_msgs)
            reordered.append(msg)
            reordered.extend(tool_msgs)
            i = j
        else:
            reordered.append(msg)
            i += 1
    messages = reordered

    return messages, tools, tool_choice


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
    messages, tools, tool_choice = extract_messages(req_data)
    # 强制使用 DeepSeek 模型，忽略客户端发送的模型名
    effective_model = DEEPSEEK_MODEL
    response_id = f"resp_{uuid.uuid4().hex[:12]}"

    if DEEPSEEK_DEBUG:
        debug_path = request.path
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n--- [{__import__('datetime').datetime.now()}] PATH={debug_path} ---\n")
            f.write(f"Request body:\n{json.dumps(req_data, indent=2, ensure_ascii=False)}\n")
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

        # response.created
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

        # response.in_progress
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

        # 构建 DeepSeek 请求
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": effective_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "thinking": {"type": "disabled"},
        }
        if tools:
            payload["tools"] = tools
            # 仅在非默认值时传递 tool_choice，DeepSeek 可能不支持此参数
            if tool_choice != "auto":
                payload["tool_choice"] = tool_choice

        # 状态跟踪
        text_item_id = f"item_{uuid.uuid4().hex[:12]}"
        full_text = ""
        full_reasoning = ""
        has_text = False
        text_started = False

        # 工具调用累积: index → {id, name, arguments, item_id, started}
        tool_calls_acc = {}

        input_tokens = 0
        output_tokens = 0
        seq = 0

        upstream = None
        try:
            upstream = _HTTP_SESSION.post(
                DEEPSEEK_URL, headers=headers, json=payload,
                stream=True, timeout=120,
            )
            upstream.raise_for_status()
            for line in upstream.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

                if "error" in chunk:
                    err = chunk["error"]
                    raise Exception(f"DeepSeek API error: {err.get('message', str(err))}")

                if "choices" not in chunk or not chunk["choices"]:
                    continue

                delta = chunk["choices"][0].get("delta", {})

                # ---- 捕获 reasoning_content（DeepSeek 思考模式必须回传）----
                reasoning_delta = delta.get("reasoning_content", "")
                if reasoning_delta:
                    full_reasoning += reasoning_delta

                # ---- 处理文本内容 ----
                content = delta.get("content", "")
                if content:
                    if not text_started:
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
                    full_text += content
                    seq += 1
                    yield "event: response.output_text.delta\n"
                    yield (
                            "data: "
                            + json.dumps({
                        "type": "response.output_text.delta",
                        "delta": content,
                        "item_id": text_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": seq,
                    }, ensure_ascii=False)
                            + "\n\n"
                    )

                # ---- 处理工具调用 ----
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_acc:
                        item_id = f"item_{uuid.uuid4().hex[:12]}"
                        tool_calls_acc[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                            "item_id": item_id,
                            "started": False,
                        }

                    acc = tool_calls_acc[idx]
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    func = tc.get("function", {})
                    if func.get("name"):
                        acc["name"] = func["name"]

                    args_delta = func.get("arguments", "")
                    if args_delta:
                        acc["arguments"] += args_delta
                        out_idx = (1 if has_text else 0) + sorted(tool_calls_acc.keys()).index(idx)

                        if not acc["started"]:
                            acc["started"] = True
                            yield "event: response.output_item.added\n"
                            yield (
                                    "data: "
                                    + json.dumps({
                                "type": "response.output_item.added",
                                "output_index": out_idx,
                                "item": {
                                    "id": acc["item_id"],
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "call_id": acc["id"],
                                    "name": acc["name"],
                                    "arguments": "",
                                },
                            }, ensure_ascii=False)
                                    + "\n\n"
                            )

                        yield "event: response.function_call_arguments.delta\n"
                        yield (
                                "data: "
                                + json.dumps({
                            "type": "response.function_call_arguments.delta",
                            "item_id": acc["item_id"],
                            "output_index": out_idx,
                            "delta": args_delta,
                        }, ensure_ascii=False)
                                + "\n\n"
                        )

            # ===== 流结束后发出完成事件 =====

            # 文本完成
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
                text_output_item = {
                    "id": text_item_id, "type": "message",
                    "status": "completed", "role": "assistant",
                    "content": [{"type": "text", "text": full_text}],
                }
                if full_reasoning:
                    text_output_item["reasoning_content"] = full_reasoning
                yield "event: response.output_item.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": text_output_item,
                }, ensure_ascii=False)
                        + "\n\n"
                )

            # 工具调用完成
            output_items = []
            if has_text:
                output_items.append({
                    "id": text_item_id, "type": "message",
                    "status": "completed", "role": "assistant",
                    "content": [{"type": "text", "text": full_text}],
                    **({"reasoning_content": full_reasoning} if full_reasoning else {}),
                })

            for idx in sorted(tool_calls_acc.keys()):
                acc = tool_calls_acc[idx]
                out_idx = (1 if has_text else 0) + sorted(tool_calls_acc.keys()).index(idx)

                yield "event: response.function_call_arguments.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.function_call_arguments.done",
                    "item_id": acc["item_id"],
                    "output_index": out_idx,
                    "arguments": acc["arguments"],
                }, ensure_ascii=False)
                        + "\n\n"
                )

                func_item = {
                    "id": acc["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": acc["id"],
                    "name": acc["name"],
                    "arguments": acc["arguments"],
                }
                if full_reasoning:
                    func_item["reasoning_content"] = full_reasoning
                yield "event: response.output_item.done\n"
                yield (
                        "data: "
                        + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": out_idx,
                    "item": func_item,
                }, ensure_ascii=False)
                        + "\n\n"
                )

                output_items.append({
                    "id": acc["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": acc["id"],
                    "name": acc["name"],
                    "arguments": acc["arguments"],
                    **({"reasoning_content": full_reasoning} if full_reasoning else {}),
                })

            # response.completed
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
            # 使用 upstream 直接读取错误响应体（流式请求下 e.response.text 可能为空）
            body = ""
            try:
                if upstream is not None:
                    body = upstream.text[:2000]
            except Exception:
                body = "(unable to read error body)"
            err_msg = f"DeepSeek API {e.response.status_code}: {body}"
            if DEEPSEEK_DEBUG:
                with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                    f.write(f"ERROR: {err_msg}\n")
                    f.write(f"Payload sent (tools={len(tools)}, msgs={len(messages)}):\n")
                    # 分别记录消息和工具，避免截断关键信息
                    payload_copy = dict(payload)
                    payload_copy.pop("messages", None)
                    payload_copy.pop("tools", None)
                    f.write(json.dumps(payload_copy, indent=2, ensure_ascii=False) + "\n")
                    f.write(f"Messages ({len(messages)}):\n")
                    f.write(json.dumps(messages, indent=2, ensure_ascii=False)[:3000] + "\n")
                    f.write(f"Tools ({len(tools)}):\n")
                    tools_str = json.dumps(tools, indent=2, ensure_ascii=False)
                    f.write(tools_str[:5000] + ("...(truncated)" if len(tools_str) > 5000 else "") + "\n")
                    total_size = len(json.dumps(payload, ensure_ascii=False))
                    f.write(f"Total payload size: {total_size} bytes ({total_size / 1024:.1f} KB)\n")
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
    globals()["DEEPSEEK_API_KEY"] = key

    from waitress import serve

    print("codex_proxy starting ...")
    print(f"   Endpoint: http://127.0.0.1:{PORT}")
    print(f"   Model:    {DEEPSEEK_MODEL}")
    print(f"   Key:      {source}")
    print(f"   Debug:    {'ON' if DEEPSEEK_DEBUG else 'OFF'}")
    print(f"   Routes:   /responses, /v1/responses, /v1/chat/completions")
    serve(app, host="127.0.0.1", port=PORT, threads=100, channel_timeout=300, cleanup_interval=30)
