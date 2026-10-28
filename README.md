# Bionic Lite 使用手册

基于对 Bionic 1.1.8（LM Studio 品牌包装，Electron + llama.cpp）逆向分析实现的轻量克隆：本地 LLM 多后端智能调度网关。零第三方依赖，仅 Python 标准库。

- 统一 OpenAI 兼容入口：`http://127.0.0.1:1235/v1`
- 自动调用本地已下载模型（Bionic/LM Studio 模型库）
- 关键词正则路由：按请求内容自动选择最适模型
- 模型热切换：驻留模型不符时自动 `lms unload → load`（对齐 Bionic 单驻留约束）
- content 回填：Bionic 部分模型强制思考模式导致正文恒空时，自动用 reasoning_content 回填
- Agent 执行：`/v1/agent` 端点，按需注入工具 + 短循环编排，可端到端完成本地文件任务

---

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.8+（实测 3.11） |
| Bionic / LM Studio | 已安装且本地服务运行（监听 `localhost:1234`） |
| lms CLI | `C:\Users\Administrator\.lmstudio\bin\lms.exe`（config.json 可改） |
| 模型 | 已在 Bionic 模型库索引（`GET /v1/models` 可见） |

检查 Bionic 服务是否就绪：`Get-NetTCPConnection -LocalPort 1234 -State Listen`

## 2. 文件说明

| 文件 | 作用 |
|---|---|
| `bionic_lite.py` | 网关主程序（零依赖，单文件） |
| `config.json` | 用户配置（端口 / 后端 / 路由规则 / 模型目录） |
| `README.md` | 本手册 |

## 3. 命令

```powershell
# 生成配置模板（已存在则跳过）
python bionic_lite.py init --config config.json

# 扫描模型目录，列出本地 GGUF（用于 /v1/models 兜底列表）
python bionic_lite.py models --config config.json

# 检查后端连通性
python bionic_lite.py status --config config.json

# 前台启动网关
python bionic_lite.py serve --config config.json

# 后台启动
Start-Process -FilePath python -ArgumentList "bionic_lite.py","serve","--config","config.json" -WorkingDirectory (Get-Location) -WindowStyle Hidden
```

查看端口确认：`Get-NetTCPConnection -LocalPort 1235 -State Listen`

停止：`Stop-Process -Id <PID>`（PID 来自上一条命令的 OwningProcess）

## 4. config.json 字段说明

| 字段 | 说明 | 默认 |
|---|---|---|
| `host` | 监听地址 | `127.0.0.1` |
| `port` | 网关端口 | `1235` |
| `default_model` | 未命中路由规则时的默认模型 | `ornith-1.5-9b` |
| `lms_cli` | lms CLI 路径 | `C:\Users\Administrator\.lmstudio\bin\lms.exe` |
| `models_dir` | 本地模型目录（GGUF 扫描） | `D:\Model\lmstudio_model` |
| `load_context` | 热切换加载上下文长度 | `8192` |
| `load_gpu` | 加载显存策略 | `max` |
| `agent_max_steps` | Agent Loop 最大工具步数 | `3` |
| `agent_tool_limit` | 单请求最多注入工具数 | `3` |
| `backends` | 后端列表（见下表） | 本地 lmstudio |
| `rules` | 路由规则（正则 → 模型名） | 4 条内置 |

### backends 条目

| 字段 | 说明 |
|---|---|
| `name` | 后端名（唯一） |
| `type` | `lmstudio`（本地）/ `openai`（任意 OpenAI 兼容）/ `vllm`（云端） |
| `base_url` | 后端 OpenAI 兼容地址，如 `http://127.0.0.1:1234/v1` |
| `api_key` | 认证 Key（本地可填 `lm-studio`） |

### rules 条目

`{"pattern": "正则", "model": "目标模型 id"}`，按请求文本正则命中改写 `model`；未命中回落到请求里的 model（缺省用 `default_model`）。

**注意：目标模型 id 必须与 `GET /v1/models` 返回的 id 完全一致**，否则热切换会快速 502。

## 5. API 使用

客户端（Cherry Studio / 脚本 / lm-evaluation-harness / 任意 OpenAI SDK）把 base_url 设为：

```
http://127.0.0.1:1235/v1
```

### 5.1 列出可用模型

```
GET http://127.0.0.1:1235/v1/models
```

后端就绪时透传 Bionic 模型列表；后端不可达时回落到本地 GGUF 索引。

### 5.2 对话（非流式）

```bash
curl -X POST http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"写一段冒泡排序"}]}'
```

### 5.3 对话（流式 SSE）

```bash
curl -N -X POST http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ornith-1.5-9b","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

### 5.4 不指定模型（自动分流）

```bash
curl -X POST http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"把这段翻译成英文：今天天气很好"}]}'
```

命中「翻译」规则 → 自动切到 gemma-4-12b-it。

### 5.5 向量（Embeddings）

```
POST http://127.0.0.1:1235/v1/embeddings
```

## 6. Agent 执行能力

网关内置轻量 Agent Loop：模型↔工具循环编排，支持本地文件类任务的端到端执行（查文件→读→总结）。零第三方依赖，通过 OpenAI tools 协议驱动本地模型。

### 6.1 端点

```
POST http://127.0.0.1:1235/v1/agent
POST http://127.0.0.1:1235/v1/chat/completions   # body 带 "agent": true 时同样走 Agent Loop
```

请求体与 chat/completions 一致（messages / model；stream 忽略，始终非流式返回；可选 `max_steps` 覆盖默认 3）。

```bash
curl -X POST http://127.0.0.1:1235/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"查找 D:\\docs 下文件名含 报告 的文件并总结"}]}'
```

除 curl 外，内置两种交互式调用方式：

**CLI 对话（`chat` 子命令）**

```bash
python bionic_lite.py chat --config config.json
```

启动网关线程并进入命令行 REPL，直接输入问题即可（自动走 Agent Loop，工具可用），`/exit` 退出。

**网页对话（`web` 子命令 / 网关根路径）**

```bash
python bionic_lite.py web --config config.json   # 自动打开浏览器
# 或先 python bionic_lite.py serve，再手动访问 http://127.0.0.1:1235/
```

网关根路径 `/` 内置零依赖聊天页（标准库，无前端构建）：模型下拉可选、消息历史保留最近 24 条、通过 `/v1/agent` 执行 Agent 对话。每条回复下方自动显示调试条：模型 / 上下文 tokens / 输出 tokens / tok/s / 耗时 / 工具步数 / 注入工具 / 工具轨迹（每步工具名+参数+结果摘要+耗时）。

**调试字段（`agent_meta`）**：`/v1/agent` 响应顶层附加 `agent_meta` 字段（非 OpenAI 标准，仅供调试）：

```json
{
  "model": "ornith-1.5-9b",
  "steps": 1,
  "tools_injected": ["list_files", "read_file", "search_files"],
  "trace": [{"tool": "list_files", "args": {"path": "D:\\code\\Marvis"}, "result": "...", "elapsed": 0.0}],
  "elapsed": 32.64,
  "tok_per_sec": 6.4,
  "context_tokens": 1183,
  "completion_tokens": 208
}
```

- `context_tokens` 即输入（prompt）token 数，可作为上下文占用观察值；`tok_per_sec` 为 completion_tokens / 总耗时 的粗粒度估算（含工具执行与热切换时间）
- 纯问答（无工具注入）时 `trace` 为空、`steps` 为 0

### 6.2 内置工具（白名单只读）

| 工具 | 说明 | 触发关键词 |
|---|---|---|
| `list_files` | 列目录（≤30 项） | 文件/目录/列出/查找… |
| `read_file` | 读文本文件（自动截断 1500 字符） | 读/查看/内容/总结… |
| `search_files` | 递归搜索文件名含关键词的文件 | 搜索/查找/找… |
| `now` | 当前时间 | 时间/日期/今天… |
| `calc` | 安全数学计算（AST 白名单） | 计算/算式… |
| `list_models` | 本地模型列表 | 模型… |

- 按用户请求关键词预选 1~3 个工具注入，纯问答不注入任何工具（零上下文开销）
- 工具执行结果统一截断到 600 字符再回填
- 工具执行器为本地白名单只读命令，不执行任意 shell

### 6.3 上下文控制（适配 8192 窗口）

- system 提示极短（固定）
- 历史裁剪：仅保留最近 6 条消息
- 单轮 max_tokens=512，最大工具步数 3（config `agent_max_steps` 可调）
- 循环内保持同一份 tools 定义，利于 KV 前缀复用；agent 循环内不触发模型热切换（避免丢 KV）

### 6.4 边界

- 9B 模型 + 8192 上下文可稳定完成 2~3 步闭环（查文件→读→总结）；大型多步任务建议换更强模型或拆分子任务
- 模型需原生支持 OpenAI tools 调用（推荐 `qwen3.5-9b` / `gemma-4-12b-it`）

## 7. 内置路由规则（可自行增删）

| 场景 | 命中示例 | 目标模型 |
|---|---|---|
| 代码 / 编程 / 调试 | "写一段 Python 排序代码" | omnicoder-9b-claude-opus-high-reasoning-distill |
| 翻译 | "把这段翻译成英文" | gemma-4-12b-it |
| 数学 / 解题 | "解这个方程" | holo-3.1-9b |
| 总结 / 问答 / 写作 | "总结这份报告" | qwen3.5-9b |
| 默认主力 | 其他 | ornith-1.5-9b |

## 8. 工作机制

1. 收到请求 → 取 messages 文本 → 正则规则匹配目标模型
2. 查询 Bionic 当前驻留模型：一致 → 直接转发；不一致 → `lms unload` → `lms load <model> -y --gpu max -c 8192` → 轮询 `/v1/models` 直到就绪（首次约 16~30s，驻留命中 5~8s）
3. 转发到后端（支持流式透传）
4. 响应中 content 为空时用 reasoning_content 回填（Bionic 强制思考模式场景）
5. `/v1/agent` 或 `agent:true` 请求：进入 Agent Loop（按需注入工具 → 循环执行 → 回填截断结果 → 最多 3 步）

## 9. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `502 模型 X 不在后端模型列表中` | 模型 id 与 `/v1/models` 不一致；先在 Bionic 导入模型并抄写准确 id |
| `503 无可用后端` | backends 配置为空或 host/port 配错 |
| 切换很慢（16~30s） | 正常，Bionic 需卸载+加载大模型；驻留命中后为秒级 |
| 流式请求被 curl 判超时 | SSE 长连接正常行为，客户端主动断开即可；应用端不受影响 |
| content 空但有 reasoning_content | 已自动回填；如需思考原文，读 `reasoning_content` 字段 |
| 本地有 GGUF 但 /v1/models 里没有 | 该文件未被 Bionic 索引，先在 Bionic 模型库扫描导入 |

## 10. 边界说明

- Bionic Lite 是调度网关，推理由 Bionic（llama.cpp）执行，不内置推理引擎
- 单实例约束与 Bionic 一致：任何时刻仅一个模型驻留显存
- 支持多后端扩展：在 backends 中加一条 `openai` 类型即可把请求转发到远程 OpenAI 兼容服务（规则里 model 用该后端可用模型 id）
