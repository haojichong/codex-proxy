# AI Proxy for Codex

<div align="center">

**让 Codex 接入国产模型**

DeepSeek · 智谱 GLM · 阿里云百炼

</div>

---

## 这是什么

Codex IDE 使用 OpenAI Responses API 协议与模型通信。国内大部分模型厂商不支持该协议，导致 Codex 无法直接接入。

本项目是一个**协议转换代理**，部署在本地，将 Codex 发出的请求实时翻译为目标模型的 API 格式，再把响应翻译回来。对 Codex 来说，它就像在和 OpenAI 通信；对模型厂商来说，收到的是标准格式的请求。

```
Codex IDE ──Responses API──▸ 本地代理 ──转换──▸ DeepSeek / GLM / 阿里云
Codex IDE ◂──Responses API── 本地代理 ◂──转换── DeepSeek / GLM / 阿里云
```

## 支持哪些模型

| 模型          | 厂商 | 后端协议 | 端口 | 获取 API Key |
|-------------|------|----------|------|-------------|
| DeepSeek V4 | DeepSeek | OpenAI Chat Completions | 5000 | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| GLM-5       | 智谱 AI | Anthropic Messages | 5001 | [open.bigmodel.cn](https://open.bigmodel.cn/) |
| GLM-5/Qwen  | 阿里云百炼 | Anthropic Messages | 5002 | 阿里云百炼控制台 |

## 功能亮点

- **协议双向转换** — OpenAI Responses API ↔ OpenAI Chat / Anthropic Messages
- **SSE 兼容** — 兼容 `data: {...}` 和 `data:{...}` 两种 SSE 格式
- **统一配置** — 所有模型配置集中在一个 `config.json`

## 快速开始

### 环境要求

- **Python >= 3.10**
- API Key（对应你想使用的模型）

### 安装

```bash
git clone https://github.com/haojichong/codex_proxy.git
cd codex_proxy
pip install -r requirements.txt
```

### 配置

编辑 `config.json`，填入你的 API Key：

```json
{
  "deepseek": {
    "api_key": "sk-your-key",
    "model": "deepseek-v4-flash",
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "port": 5000,
    "debug": false
  },
  "glm": {
    "api_key": "your-key",
    "model": "glm-5",
    "base_url": "https://open.bigmodel.cn/api/anthropic",
    "auth_type": "x-api-key",
    "port": 5001,
    "debug": false
  },
  "aliyun": {
    "api_key": "sk-your-key",
    "model": "glm-5",
    "base_url": "https://coding.dashscope.aliyuncs.com/apps/anthropic",
    "auth_type": "bearer",
    "port": 5002,
    "debug": false
  }
}
```

### 启动

**菜单选择：**

```bash
start.bat
```

**或直接启动：**

```bash
python deepseek_proxy.py   # DeepSeek (端口 5000)
python glm_proxy.py        # 智谱 GLM (端口 5001)
python aliyun_proxy.py     # 阿里云百炼 (端口 5002)
```

## 配置参考

### 通用字段

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | 是 | — | API 密钥，未配置时启动会提示输入 |
| `model` | 是 | — | 模型名称 |
| `port` | 否 | 5000/5001/5002 | 本地监听端口 |
| `debug` | 否 | `false` | 设为 `true` 写调试日志到 `debug_*.log` |

### DeepSeek 专用

| 字段 | 说明 |
|------|------|
| `api_url` | API 完整地址 |

### GLM / 阿里云专用

| 字段 | 说明 |
|------|------|
| `base_url` | Anthropic 兼容端点（不含 `/v1/messages`） |
| `auth_type` | `x-api-key`（智谱）或 `bearer`（阿里云） |

## Codex + cc-switch 配置

[cc-switch](https://github.com/nicepkg/cc-switch) 是 Codex IDE 的模型切换插件，配合本代理可以自由选择模型。

### 配置步骤

1. 在 Codex IDE 中安装 cc-switch 插件

2. 启动你想使用的代理，例如 DeepSeek：
   ```bash
   (venv) PS > ./start.bat
   
   +============================================+
   |        AI Proxy Service Manager            |
   +============================================+
   |  [1] DeepSeek Proxy  (Port 5000)           |
   |  [2] GLM-5 Proxy     (Port 5001)           |
   |  [3] Aliyun Bailian  (Port 5002)           |
   |  [Q] Exit                                  |
   +============================================+
   
   Select [1/2/3/Q]: 2
   
   [GLM-5] Starting...
   glm_proxy starting ...
   Endpoint: http://127.0.0.1:5001
   Model:    glm-5.1
   API URL:  https://open.bigmodel.cn/api/anthropic/v1/messages
   Key:      config.json
   Debug:    OFF
   Routes:   /responses, /v1/responses, /v1/chat/completions
   ```

3. 在 cc-switch 中将模型地址设为本地代理，测试成功后就可以在Codex中使用国产模型对话了：

   | 模型 | 地址 |
   |------|------|
   | DeepSeek | `http://127.0.0.1:5000` |
   | 智谱 GLM | `http://127.0.0.1:5001` |
   | 阿里云百炼 | `http://127.0.0.1:5002` |
![cc-switch.jpg](image%2Fcc-switch.jpg)

