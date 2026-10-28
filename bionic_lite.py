#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bionic Lite — 本地 LLM 多后端智能调度网关（Bionic / LM Studio 克隆）
========================================================================
基于对 Bionic 1.1.8 (LM Studio 品牌包装) 的逆向分析实现的核心能力子集：

  1. OpenAI 兼容统一入口（默认 127.0.0.1:1235）：/v1/models、/v1/chat/completions(流式+非流式)、/v1/embeddings
  2. 多后端适配：lmstudio(本地 1234) / openai(任意 OpenAI 兼容远程) / vllm(云端)
  3. 关键词正则路由：按请求文本或模型名自动选后端与模型，未命中回落到默认主力
  4. 模型热切换：lmstudio 后端驻留模型不符时，自动 lms unload / load（对齐 Bionic 单驻留约束）
  5. 模型索引：扫描 models 目录 GGUF 文件生成可用模型清单
  6. content 回填：后端强制思考模式导致 content 恒空时，自动用 reasoning_content 回填

零第三方依赖（仅 Python 标准库）。运行：python bionic_lite.py serve
"""

import argparse
import ast
import datetime
import io
import json
import operator
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 1235,
    "default_model": "ornith-1.5-9b",
    "lms_cli": r"C:\Users\Administrator\.lmstudio\bin\lms.exe",
    "models_dir": r"D:\Model\lmstudio_model",
    "load_context": 8192,
    "load_gpu": "max",
    "agent_max_steps": 3,
    "agent_tool_limit": 3,
    "backends": [
        {
            "name": "local",
            "type": "lmstudio",
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key": "lm-studio"
        }
    ],
    "rules": [
        {"pattern": r"(?i)(代码|编程|debug|bug|refactor|函数|class|python|javascript|c\\+\\+|typescript|代码审查|编写|实现)", "model": "omnicoder-9b-claude-opus-high-reasoning-distill"},
        {"pattern": r"(?i)(翻译|translate|英译|中译|英文|中文)", "model": "gemma-4-12b-it"},
        {"pattern": r"(?i)(数学|计算|方程|导数|积分|解题|solve|math|概率|统计)", "model": "holo-3.1-9b"},
        {"pattern": r"(?i)(格式化|json|结构化|表格|总结|摘要|问答|写作|报告|邮件|文案|大纲)", "model": "qwen3.5-9b"}
    ]
}


class ModelIndex:
    """扫描 models_dir 下的 GGUF 模型文件，建立 文件名 -> 模型名 索引。"""

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.index = {}

    def scan(self):
        self.index = {}
        if not self.models_dir.exists():
            return self.index
        for p in sorted(self.models_dir.rglob("*.gguf")):
            name = p.stem  # 去掉 .gguf，作为候选模型名
            self.index[name] = str(p)
        return self.index

    def list_models(self):
        return sorted(self.index.keys())


class Router:
    """关键词正则路由 + 后端选择。"""

    def __init__(self, config):
        self.rules = []
        for r in config.get("rules", []):
            try:
                self.rules.append((re.compile(r["pattern"]), r["model"]))
            except re.error as e:
                print(f"[warn] 跳过非法正则 {r.get('pattern')!r}: {e}")

    def route(self, requested_model, text):
        for rx, model in self.rules:
            if rx.search(text or ""):
                return model
        return requested_model or None


class Backend:
    """后端抽象：统一 forward(方法, 路径, body, headers, stream)。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.base = cfg["base_url"].rstrip("/")
        self.key = cfg.get("api_key", "")

    def _headers(self, extra=None):
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        if extra:
            h.update(extra)
        return h

    def request(self, method, path, body=None, timeout=600, stream=False):
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            return e.code, io.BytesIO(e.read())
        return resp.status, resp

    def is_ready(self):
        try:
            status, _ = self.request("GET", "/models", timeout=5)
            return status == 200
        except Exception:
            return False


class LmStudioBackend(Backend):
    """本地 Bionic/LM Studio：支持驻留模型热切换。"""

    def _model_ids(self):
        """拉取后端可用的完整模型 id 列表（含已索引未驻留的）。"""
        try:
            status, resp = self.request("GET", "/models", timeout=10)
            if status != 200:
                return []
            data = json.loads(resp.read().decode())
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def loaded_model(self):
        """查询当前驻留模型名（通过 /v1/models 找 id）。"""
        try:
            status, resp = self.request("GET", "/models", timeout=10)
            if status != 200:
                return None
            data = json.loads(resp.read().decode())
            ids = [m["id"] for m in data.get("data", [])]
            return ids[0] if ids else None
        except Exception:
            return None

    def _match(self, current, target):
        """宽松匹配：完全相等，或目标为驻留模型前缀（如 omnicoder-9b 命中 omnicoder-9b-claude-...）。"""
        if not current:
            return False
        if current == target:
            return True
        return current.startswith(target) or target.startswith(current)

    def hot_switch(self, target, lms_cli, context, gpu):
        """驻留模型不符时，lms unload -> load，轮询就绪。与 Bionic 单驻留约束对齐。"""
        current = self.loaded_model()
        if self._match(current, target):
            return True, "already-loaded"
        known = self._model_ids()
        if known and target not in known and not any(self._match(k, target) for k in known):
            return False, f"模型 {target} 不在后端模型列表中（可用: /v1/models）"
        if not Path(lms_cli).exists():
            return False, f"lms CLI 不存在: {lms_cli}"
        try:
            subprocess.run([lms_cli, "unload"], capture_output=True, timeout=60)
            subprocess.run([lms_cli, "load", target, "-y", "--gpu", gpu, "-c", str(context)],
                           capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            return False, "lms 切换超时"
        # 轮询就绪（对齐 Bionic 首次切换 16~30s）
        for _ in range(120):
            if self._match(self.loaded_model(), target):
                return True, "switched"
            time.sleep(1)
        return False, f"切换后 {target} 未就绪"


# ============================================================
# Agent 执行能力：工具注册表 + 短循环编排（零第三方依赖）
# ============================================================

AGENT_SYSTEM = (
    "你是本地 Agent。可调用工具完成用户任务。"
    "工具包括：list_files/read_file/search_files（文件）、now/calc（时间计算）、"
    "list_models（模型）、shell（执行本机 PowerShell 命令，仅查询类）。"
    "工具结果可能被截断，只依据返回内容作答；无法确定时如实说明。"
    "回答精炼，用中文。"
)


class ToolRegistry:
    """本地工具注册表：按任务关键词按需注入 1~3 个工具，白名单只读执行。"""

    RESULT_LIMIT = 600  # 工具结果回填上限（字符）

    def __init__(self, gw):
        self.gw = gw
        self.tools = self._build_tools()

    # ---------- 工具实现（白名单，只读/无害） ----------
    @staticmethod
    def _list_files(path="."):
        p = Path(path)
        if not p.exists():
            return f"路径不存在: {path}"
        if p.is_file():
            return f"{p.name} (文件, {p.stat().st_size} 字节)"
        items = []
        try:
            for child in sorted(p.iterdir())[:30]:
                try:
                    if child.is_dir():
                        items.append(f"[D] {child.name}/")
                    else:
                        items.append(f"[F] {child.name} ({child.stat().st_size}B)")
                except OSError:
                    continue
        except OSError:
            return f"无法读取目录: {path}"
        return "\n".join(items) if items else f"(空目录: {path})"

    @staticmethod
    def _read_file(path, max_chars=1500):
        p = Path(path)
        if not p.exists():
            return f"文件不存在: {path}"
        if p.is_dir():
            return "(是目录，请改用 list_files)"
        try:
            data = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"读取失败: {e}"
        if len(data) > max_chars:
            return data[:max_chars] + f"\n...(已截断，共 {len(data)} 字符)"
        return data

    @staticmethod
    def _search_files(root=".", keyword="", max_results=10):
        base = Path(root)
        if not base.exists():
            return f"目录不存在: {root}"
        kw = (keyword or "").lower()
        found = []
        try:
            for p in base.rglob("*"):
                if len(found) >= max_results:
                    break
                if kw in p.name.lower():
                    found.append(str(p))
        except OSError:
            pass
        return "\n".join(found) if found else f"未找到名称含 '{keyword}' 的文件"

    @staticmethod
    def _now():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

    @staticmethod
    def _calc(expr):
        tree = ast.parse(expr, mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                   ast.USub, ast.UAdd)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError("表达式包含不允许的操作")
        return str(eval(compile(tree, "<expr>", "eval"),
                        {"__builtins__": {}}, {}))

    def _list_models(self):
        return "\n".join(self.gw.index.list_models()) or "(索引为空)"

    _SHELL_BLOCKED = [
        "format-volume", "format /q", "format c:", "format d:", "format e:", "format f:",
        "diskpart", "bcdedit", "bootrec",
        "del /", "rmdir /s", "rd /s", "remove-item", "rm -rf", "clear-content",
        "shutdown", "restart-computer", "stop-computer",
        "reg delete", "reg add", "set-itemproperty", "remove-itemproperty",
        "stop-service", "sc delete", "wmic process delete", "taskkill /f /im explorer",
    ]

    @staticmethod
    def _exec_shell(command):
        import subprocess
        low = (command or "").lower()
        hit = next((b for b in ToolRegistry._SHELL_BLOCKED if b in low), None)
        if hit:
            return f"已拒绝：命令含破坏性操作（命中黑名单项: {hit}），网关不允许执行。"
        if len(command) > 500:
            return "已拒绝：命令过长（>500 字符）。"
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            out = (r.stdout or "").strip()
            if not out:
                out = (r.stderr or "").strip()
            if r.returncode != 0 and not out:
                return f"命令退出码 {r.returncode}（无输出）"
            return out or "(无输出)"
        except subprocess.TimeoutExpired:
            return "命令超时（15 秒），已终止。"
        except Exception as e:
            return f"执行失败: {e}"

    def _build_tools(self):
        return [
            {
                "name": "shell",
                "description": "在本机执行一条 PowerShell 命令并返回输出。可用于查询系统信息、进程、磁盘、网络、服务、环境变量等。破坏性命令（删除/格式化/关机/改注册表/停服务）会被网关拒绝。",
                "params": {"command": {"type": "string", "description": "要执行的 PowerShell 命令，如 Get-Process、systeminfo、Get-PSDrive"}},
                "required": ["command"],
                "keywords": ["命令", "执行", "运行", "shell", "cmd", "powershell", "终端", "命令行", "系统信息", "进程", "磁盘", "硬盘", "分区", "硬件", "网络", "服务", "ipconfig", "systeminfo", "环境变量"],
                "fn": self._exec_shell,
            },
            {
                "name": "list_files",
                "description": "列出目录下的文件与子目录（最多 30 项）",
                "params": {"path": {"type": "string", "description": "目录绝对路径，缺省为当前目录"}},
                "required": [],
                "keywords": ["文件", "目录", "文件夹", "列出", "查找", "搜索", "list", "dir", "folder", "目录下"],
                "fn": self._list_files,
            },
            {
                "name": "read_file",
                "description": "读取文本文件内容（UTF-8，超长自动截断到 max_chars）",
                "params": {
                    "path": {"type": "string", "description": "文件绝对路径"},
                    "max_chars": {"type": "integer", "description": "最多读取字符数，缺省 1500"},
                },
                "required": ["path"],
                "keywords": ["读", "查看", "内容", "文件", "打开", "read", "cat", "看下", "总结"],
                "fn": self._read_file,
            },
            {
                "name": "search_files",
                "description": "在目录下递归搜索文件名包含关键词的文件（最多 max_results 个）",
                "params": {
                    "root": {"type": "string", "description": "搜索根目录绝对路径"},
                    "keyword": {"type": "string", "description": "文件名关键词"},
                    "max_results": {"type": "integer", "description": "结果上限，缺省 10"},
                },
                "required": ["keyword"],
                "keywords": ["搜索", "查找", "找", "文件", "在哪", "哪个文件", "search", "find", "定位"],
                "fn": self._search_files,
            },
            {
                "name": "now",
                "description": "获取当前日期与时间",
                "params": {},
                "required": [],
                "keywords": ["时间", "日期", "今天", "现在", "几点", "now", "date", "time"],
                "fn": self._now,
            },
            {
                "name": "calc",
                "description": "执行安全数学计算（仅支持数字、四则运算、幂、取模与括号）",
                "params": {"expr": {"type": "string", "description": "数学表达式，如 2**10"}},
                "required": ["expr"],
                "keywords": ["计算", "算式", "等于", "math", "calc", "运算", "多少"],
                "fn": self._calc,
            },
            {
                "name": "list_models",
                "description": "列出本地索引到的可用模型名",
                "params": {},
                "required": [],
                "keywords": ["模型", "model", "有哪些模型", "本地模型"],
                "fn": self._list_models,
            },
        ]

    # ---------- 按需注入 ----------
    def match_schemas(self, text, limit=3):
        """按用户文本关键词预选工具，返回 OpenAI function schema 列表。"""
        low = (text or "").lower()
        matched = [t for t in self.tools if any(k.lower() in low for k in t["keywords"])]
        if not matched:
            return []  # 纯问答，不注入任何工具，省上下文
        return [self._schema(t) for t in matched[:limit]]

    @staticmethod
    def _schema(t):
        return {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {"type": "object",
                               "properties": t["params"],
                               "required": t["required"]},
            },
        }

    # ---------- 执行 ----------
    def call(self, name, args):
        for t in self.tools:
            if t["name"] == name:
                try:
                    result = t["fn"](**args)
                except Exception as e:
                    return f"工具执行失败: {e}"
                return self._truncate(str(result))
        return f"未知工具: {name}"

    @classmethod
    def _truncate(cls, text, limit=None):
        limit = limit or cls.RESULT_LIMIT
        if len(text) > limit:
            return text[:limit] + f"\n...(结果已截断，共 {len(text)} 字符)"
        return text


class AgentLoop:
    """模型↔工具短循环编排：历史裁剪 + 按需注入 + 结果截断 + 最大步数。"""

    MAX_HISTORY = 6      # 保留最近消息条数（不含 system）
    DEFAULT_STEPS = 3    # 最大工具循环步数
    MAX_TOKENS = 512     # 单轮生成上限

    @staticmethod
    def _msg_text(m):
        """多模态兼容：content 可能是 str，也可能是 OpenAI 多模态 list（text / image_url）。"""
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            out = []
            for part in c:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        out.append(str(part["text"]))
                    elif part.get("type") == "image_url":
                        out.append("[图片]")
            return " ".join(out)
        return str(c)

    def __init__(self, gw, backend, target, max_steps=None):
        self.gw = gw
        self.backend = backend
        self.target = target
        self.max_steps = max_steps or gw.config.get("agent_max_steps", self.DEFAULT_STEPS)
        self.trace = []
        self.tool_names = []
        self._t0 = time.time()

    def run(self, messages, extra_kwargs=None, on_event=None):
        def emit(ev):
            if on_event:
                try:
                    on_event(ev)
                except Exception:
                    pass

        msgs = self._trim_history(messages)
        text = " ".join(self._msg_text(m) for m in messages
                        if isinstance(m, dict) and m.get("role") == "user")
        tool_defs = self.gw.tools.match_schemas(text)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        sys_msg = {"role": "system", "content": AGENT_SYSTEM}
        msgs = [sys_msg] + msgs
        body = {"model": self.target, "messages": msgs,
                "max_tokens": self.MAX_TOKENS, "stream": False}
        if extra_kwargs:
            body.update(extra_kwargs)  # 透传客户端附加参数（如 chat_template_kwargs 关闭思考）
        if tool_defs:
            body["tools"] = tool_defs  # 循环内保持同一份 tools，利于 KV 前缀复用
            self.tool_names = [t.get("function", {}).get("name", "") for t in tool_defs]

        for step in range(self.max_steps):
            emit({"type": "stage", "stage": "infer", "step": step + 1, "model": self.target})
            try:
                status, resp = self.backend.request("POST", "/chat/completions", body, timeout=600)
                raw = resp.read()
            except urllib.error.HTTPError as e:
                return self._finish_error(f"后端错误: {e.code} {e.read().decode(errors='ignore')[:300]}")
            except Exception as e:
                return self._finish_error(f"后端不可达: {e}")
            if status != 200:
                detail = raw.decode(errors="ignore").strip()[:300]
                return self._finish_error(f"后端错误 {status}: {detail or '(无响应体)'}")
            try:
                obj = json.loads(raw.decode())
            except Exception as e:
                return self._finish_error(f"后端响应解析失败: {e}")

            ch = obj.get("choices", [{}])[0]
            msg = ch.get("message", {})
            self._accumulate(usage, obj.get("usage", {}))
            emit({"type": "usage", "prompt_tokens": usage["prompt_tokens"],
                  "completion_tokens": usage["completion_tokens"]})

            if msg.get("tool_calls"):
                msgs.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": msg["tool_calls"]})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    t1 = time.time()
                    result = self.gw.tools.call(name, args)
                    entry = {
                        "tool": name,
                        "args": args,
                        "result": (result or "")[:120],
                        "elapsed": round(time.time() - t1, 2),
                    }
                    self.trace.append(entry)
                    emit({"type": "tool", "step": step + 1, "tool": name, "args": args,
                          "result": (result or "")[:200], "elapsed": entry["elapsed"]})
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result})
                body["messages"] = msgs
                continue

            # 无工具调用：content 为空时用 reasoning 回填（避免纯思考模型回复为空），并移除 reasoning 字段防止前端重复展示
            if not msg.get("content") and msg.get("reasoning_content"):
                msg["content"] = msg["reasoning_content"]
                msg.pop("reasoning_content", None)
            return self._finish(msg, usage, finish_reason=ch.get("finish_reason", "stop"))

        # 超过最大步数：返回最后一条 assistant 消息或提示
        last_assistant = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if last_assistant and last_assistant.get("content"):
            return self._finish(last_assistant, usage, finish_reason="stop")
        return self._finish_error(f"达到最大工具步数 {self.max_steps}，任务未完成")

    # ---------- 内部辅助 ----------
    @staticmethod
    def _trim_history(messages):
        """裁剪历史：去掉 system，保留最后 MAX_HISTORY 条，并确保含最后一条 user。"""
        clean = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
        if not clean:
            return [{"role": "user", "content": "(空)"}]
        tail = clean[-AgentLoop.MAX_HISTORY:]
        if tail and tail[-1].get("role") != "user":
            # 兜底：确保以 user 消息结尾
            tail = tail + [{"role": "user", "content": "(请继续)"}]
        return tail

    @staticmethod
    def _accumulate(usage, u):
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
        usage["total_tokens"] += u.get("total_tokens", 0)

    def _finish(self, msg, usage, finish_reason="stop"):
        elapsed = round(time.time() - self._t0, 2)
        comp = usage.get("completion_tokens", 0)
        tps = round(comp / elapsed, 1) if elapsed > 0 else 0.0
        meta = {
            "model": self.target,
            "steps": len(self.trace),
            "tools_injected": self.tool_names,
            "trace": self.trace,
            "elapsed": elapsed,
            "tok_per_sec": tps,
            "context_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": comp,
        }
        return {
            "id": f"agent-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.target,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
            "usage": usage,
            "agent_meta": meta,
        }

    def _finish_error(self, detail):
        msg = {"role": "assistant", "content": f"[agent 执行失败] {detail}"}
        return self._finish(msg, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            finish_reason="error")


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "BionicLite/1.0"

    def _json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json(status, {"error": {"message": message, "type": "bionic_lite_error", "code": status}})

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return b"{}"
        return self.rfile.read(length)

    @staticmethod
    def _msg_text(m):
        """多模态兼容：content 可能是 str，也可能是 OpenAI 多模态 list（text / image_url）。"""
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            out = []
            for part in c:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        out.append(str(part["text"]))
                    elif part.get("type") == "image_url":
                        out.append("[图片]")
            return " ".join(out)
        return str(c)

    def _msg_text_list(self, messages):
        return " ".join(self._msg_text(m) for m in messages if isinstance(m, dict))

    # ---------- 访问令牌校验 ----------
    def _client_is_local(self):
        """回环地址（127.0.0.1 / ::1 / localhost）视为本机，免令牌。"""
        addr = (self.client_address or ("", 0))[0]
        return addr in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost")

    def _check_token(self):
        """本机免令牌；非本机请求必须携带 config.api_token（Authorization: Bearer / X-API-Token / ?token=）。"""
        gw: Gateway = self.server.gw
        token = gw.config.get("api_token")
        if not token:
            return True  # 未配置令牌则保持开放
        if self._client_is_local():
            return True
        # 优先取 header，其次 URL query
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip() == token
        if self.headers.get("X-API-Token", "") == token:
            return True
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        return q.get("token", [""])[0] == token

    def _authorize(self):
        """API 路由统一入口校验，失败返回 401。"""
        if not self._check_token():
            self._error(401, "缺少或错误的访问令牌（本机免令牌；非本机需携带 Authorization: Bearer <token> 或 X-API-Token）")
            return False
        return True

    # ---------- OpenAI 兼容路由 ----------
    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("", "/"):
            self._serve_chat_page()
            return
        if not self._authorize():
            return
        if path == "/v1/models":
            self._handle_models()
        else:
            self._error(404, f"Not Found: {self.path}")

    def do_POST(self):
        if not self._authorize():
            return
        if self.path.rstrip("/") == "/v1/chat/completions":
            self._handle_chat()
        elif self.path.rstrip("/") == "/v1/agent":
            self._handle_agent()
        elif self.path.rstrip("/") == "/v1/embeddings":
            self._handle_embedding()
        else:
            self._error(404, f"Not Found: {self.path}")

    # ---------- 各端点实现 ----------
    _CHAT_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bionic Lite · Agent 对话</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 .wrap{max-width:1180px;margin:0 auto;padding:16px;display:flex;gap:16px;align-items:flex-start}
 .main{flex:1;min-width:0}
 header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
 h1{font-size:18px;margin:0;flex:1}
 select{background:#1c1f27;color:#e6e6e6;border:1px solid #333;border-radius:6px;padding:6px 8px}
 input#model{background:#1c1f27;color:#e6e6e6;border:1px solid #333;border-radius:6px;padding:6px 8px;min-width:250px;font-size:13px}
 input#model::placeholder{color:#6a7280}
 label.nothink{display:flex;align-items:center;gap:5px;font-size:13px;color:#9aa3b2;cursor:pointer;white-space:nowrap}
 label.nothink input{accent-color:#3d6df2;width:15px;height:15px;cursor:pointer}
 #log{height:calc(100vh - 190px);overflow-y:auto;border:1px solid #2a2f3a;border-radius:10px;padding:12px;background:#14171d}
 .msg{margin:10px 0;line-height:1.55;white-space:pre-wrap}
 .msg .who{font-weight:700;margin-right:8px}
 .msg.user .who{color:#4da3ff}.msg.assistant .who{color:#4caf7d}.msg.tool .who{color:#e0a63c}
 .msg.user .b{display:inline-block;max-width:85%;text-align:left;padding:8px 12px;border-radius:10px;background:#1b2130}
 .msg.assistant .b{display:inline-block;max-width:85%;padding:8px 12px;border-radius:10px;background:#18261f}
 .msg.tool .b{background:#2a2413;font-family:monospace;font-size:12px;white-space:pre-wrap}
 .msg.err .b{display:inline-block;max-width:85%;padding:8px 12px;border-radius:10px;background:#3a1a1a;color:#ff9d9d}
 .msg .meta{font-size:11px;color:#7a8291;margin-top:5px;line-height:1.5}
 .msg .meta.trace{color:#8a7a4d;word-break:break-all}
 .msg .think{font-size:12px;color:#9aa3b2;background:#1a1e28;border-left:3px solid #4a5568;border-radius:6px;padding:6px 10px;margin-bottom:6px}
 .msg .think summary{cursor:pointer;user-select:none}
 .msg .think .tb{white-space:pre-wrap;color:#8a92a5;margin-top:4px}
 .imgrow{display:flex;align-items:center;gap:8px;margin-bottom:8px}
 .inputbar{display:flex;gap:8px;margin-top:10px}
 input[type=text]{flex:1;padding:10px 12px;border-radius:8px;border:1px solid #333;background:#1c1f27;color:#e6e6e6}
 button{padding:10px 18px;border:0;border-radius:8px;background:#3d6df2;color:#fff;cursor:pointer}
 button:disabled{opacity:.5}
 pre{background:#0d0f13;padding:8px;border-radius:6px;overflow-x:auto}
 #side{width:300px;flex:none;background:#131620;border:1px solid #2a2f3a;border-radius:10px;padding:12px;font-size:12px;position:sticky;top:16px;max-height:calc(100vh - 32px);overflow-y:auto}
 #side h2{font-size:14px;margin:0 0 10px;color:#c9d1e0}
 .sideBlock{margin:12px 0}
 .sideTitle{color:#9aa3b2;margin-bottom:6px}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#555;margin-right:6px;vertical-align:middle}
 .dot.load{background:#e0a63c;animation:pulse 1s infinite}
 .dot.infer{background:#3d6df2;animation:pulse 1s infinite}
 .dot.tool{background:#e0a63c}
 .dot.ready,.dot.done{background:#4caf7d}
 .dot.err{background:#e05a5a}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
 .bar{height:8px;background:#1c1f27;border-radius:5px;overflow:hidden}
 .barfill{height:100%;background:linear-gradient(90deg,#3d6df2,#4caf7d);transition:width .3s}
 .hint{color:#6a7280}
 .step{border-left:2px solid #3d6df2;padding:4px 6px;margin:6px 0;background:#181c26;border-radius:4px}
 .step .snum{display:inline-block;min-width:18px;color:#4da3ff;font-weight:700}
 .step .sname{color:#e0a63c;font-weight:700}
 .step .stime{float:right;color:#7a8291}
 .step .sargs,.step .sres{font-family:monospace;font-size:11px;color:#9aa3b2;margin-top:2px;word-break:break-all}
 #steps{max-height:300px;overflow-y:auto}
</style>
</head>
<body>
<div class="wrap">
 <div class="main">
  <header><h1>Bionic Lite · Agent 对话</h1><input id="model" list="modelList" placeholder="默认模型 · 输入关键字过滤" autocomplete="off"><datalist id="modelList"><option value=""></option></datalist><label class="nothink"><input type="checkbox" id="nothink"> 关闭思考</label></header>
  <div id="log"></div>
  <div class="imgrow"><input type="file" id="file" accept="*" style="display:none"><button id="imgBtn" style="background:#2a3040" title="上传图片或文本/代码附件，随消息发送">附件</button><span id="imgPrev"></span></div>
  <div class="inputbar"><input id="text" type="text" placeholder="输入问题，回车发送。可用工具：list_files / read_file / search_files / now / calc / list_models / shell" autocomplete="off"><button id="send">发送</button></div>
 </div>
 <aside id="side">
  <h2>工作状态</h2>
  <div class="sideBlock"><span class="dot idle" id="stageDot"></span><span id="stageText">空闲</span></div>
  <div class="sideBlock">
   <div class="sideTitle">上下文占用 <span id="ctxText">0 / 8192 tok</span></div>
   <div class="bar"><div id="ctxBar" class="barfill" style="width:0%"></div></div>
  </div>
  <div class="sideBlock">
   <div class="sideTitle">工具轨迹</div>
   <div id="steps"><div class="hint">暂无</div></div>
  </div>
  <div class="sideBlock">
   <div class="sideTitle">汇总</div>
   <div id="sumText" class="hint">—</div>
  </div>
 </aside>
</div>
<script>
const log=document.getElementById('log'),text=document.getElementById('text'),send=document.getElementById('send'),model=document.getElementById('model'),nothink=document.getElementById('nothink'),file=document.getElementById('file'),imgBtn=document.getElementById('imgBtn'),imgPrev=document.getElementById('imgPrev');
let history=[],pendingParts=[];
model.value=localStorage.getItem('bmodel')||'__DEFAULT_MODEL__';
model.addEventListener('input',()=>localStorage.setItem('bmodel',model.value.trim()));
const BIN_EXT=['pdf','doc','docx','xls','xlsx','ppt','pptx','zip','rar','7z','gz','tar','exe','msi','dll','so','mp3','mp4','avi','mov','iso'];
function compressImage(f){return new Promise((res,rej)=>{const fr=new FileReader();fr.onload=()=>{const im=new Image();im.onload=()=>{const MAX=1024;let w=im.width,h=im.height;if(w>MAX||h>MAX){const r=Math.min(MAX/w,MAX/h);w=Math.round(w*r);h=Math.round(h*r);}const cv=document.createElement('canvas');cv.width=w;cv.height=h;cv.getContext('2d').drawImage(im,0,0,w,h);res({type:f.type||'image/jpeg',data:cv.toDataURL('image/jpeg',0.8)});};im.onerror=()=>rej(new Error('图片解码失败'));im.src=fr.result;};fr.onerror=()=>rej(new Error('读取失败'));fr.readAsDataURL(f);});}
function renderPrev(){imgPrev.innerHTML='';pendingParts.forEach((p,i)=>{const el=document.createElement('span');el.style.cssText='display:inline-flex;align-items:center;gap:6px;margin-right:8px';if(p.kind==='image'){const im=document.createElement('img');im.src=p.data;im.style.cssText='height:44px;border-radius:6px';el.appendChild(im);}else{const lb=document.createElement('span');lb.textContent=p.name;lb.style.cssText='max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:#232836;padding:4px 8px;border-radius:6px;font-size:12px';el.appendChild(lb);}const rm=document.createElement('button');rm.textContent='✕';rm.style.cssText='padding:2px 6px;background:#5a2a2a;font-size:11px';rm.onclick=()=>{pendingParts.splice(i,1);renderPrev();};el.appendChild(rm);imgPrev.appendChild(el);});}
imgBtn.onclick=()=>file.click();
file.onchange=()=>{const f=file.files&&file.files[0];file.value='';if(!f)return;if(f.type.startsWith('image/')){compressImage(f).then(r=>{pendingParts.push({kind:'image',data:r.data,name:f.name});renderPrev();}).catch(e=>alert(e.message));return;}const ext=(f.name.split('.').pop()||'').toLowerCase();if(BIN_EXT.indexOf(ext)>=0){alert('暂不支持该二进制附件('+ext+')，可上传文本/代码类文件');return;}if(f.size>300*1024){alert('文件过大（>300KB），暂不支持');return;}const fr=new FileReader();fr.onload=()=>{let t=String(fr.result||'');if(t.length>6000)t=t.slice(0,6000)+'\\n...[内容已截断]';pendingParts.push({kind:'text',name:f.name,text:t});renderPrev();};fr.onerror=()=>alert('读取失败');fr.readAsText(f);};
function buildContent(){const parts=[];const q=text.value.trim();if(q)parts.push({type:'text',text:q});for(const p of pendingParts){if(p.kind==='image')parts.push({type:'image_url',image_url:{url:p.data}});else parts.push({type:'text',text:'【附件 '+p.name+'】\\n'+p.text});}return parts;}
function getToken(){let t=localStorage.getItem('btoken')||'';if(!t){t=prompt('本机访问无需令牌。若非本机访问，请输入访问令牌：');if(t)localStorage.setItem('btoken',t);}return t;}
function clearToken(){localStorage.removeItem('btoken');}
function authHeaders(extra){const h=Object.assign({'Content-Type':'application/json'},extra||{});const t=getToken();if(t)h['X-API-Token']=t;return h;}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function renderMd(s){
  s=esc(s);
  s=s.replace(/```([\\s\\S]*?)```/g,(m,c)=>'<pre>'+c+'</pre>');
  s=s.replace(/\\*\\*([^*]+)\\*\\*/g,'<b>$1</b>');
  s=s.replace(/\\n/g,'<br>');
  return s;
}
const whoMap={user:'你',assistant:'AI',tool:'工具',err:'错误'};
function add(who,content){
  const d=document.createElement('div');d.className='msg '+who;
  const w=document.createElement('span');w.className='who';w.textContent=whoMap[who]||who;
  const b=document.createElement('span');b.className='b';b.innerHTML=renderMd(content);
  d.appendChild(w);d.appendChild(b);log.appendChild(d);log.scrollTop=log.scrollHeight;
  return d;
}
function appendMeta(el,m){
  if(!m)return;
  const md=document.createElement('div');md.className='meta';
  md.textContent='模型 '+m.model+' · 上下文 '+m.context_tokens+' tok · 输出 '+m.completion_tokens+' tok · '+m.tok_per_sec+' tok/s · 耗时 '+m.elapsed+'s · 工具步数 '+m.steps+(m.tools_injected&&m.tools_injected.length?' · 注入 ['+m.tools_injected.join(',')+']':'');
  el.appendChild(md);
  if(m.trace&&m.trace.length){
    const tr=document.createElement('div');tr.className='meta trace';
    tr.textContent='工具轨迹: '+m.trace.map((t,i)=>i+1+'.'+t.tool+'('+JSON.stringify(t.args)+')→'+String(t.result).replace(/\s+/g,' ').slice(0,40)+(t.elapsed!=null?' ['+t.elapsed+'s]':'')).join(' ⮕ ');
    el.appendChild(tr);
  }
}
fetch('/v1/models',{headers:authHeaders()}).then(r=>r.json()).then(j=>{if(j.data){const dl=document.getElementById('modelList');for(const m of j.data){const o=document.createElement('option');o.value=m.id;o.textContent=m.id;dl.appendChild(o);}}}).catch(()=>{});
async function ask(){
  const t0=Date.now();let readyTs=null;
  const parts=buildContent();if(!parts.length){text.focus();return;}
  text.value='';send.disabled=true;send.textContent='思考中…';
  const content=parts.length===1&&parts[0].type==='text'?parts[0].text:parts;
  history.push({role:'user',content:content});
  const disp=parts.map(p=>{if(p.type==='image_url')return '[图片]';const t=p.text||'';const m=t.match(/^【附件 ([^】]+)】/);return m?'[附件 '+m[1]+']':t;}).join(' ').trim()||'[图片]';
  add('user',disp);
  pendingParts=[];renderPrev();
  resetSide();
  try{
    const payload={model:model.value||undefined,messages:history,stream:true};
    if(nothink.checked)payload.chat_template_kwargs={enable_thinking:false};
    let r=await fetch('/v1/agent',{method:'POST',headers:authHeaders(),body:JSON.stringify(payload)});
    if(r.status===401){clearToken();const t2=prompt('访问令牌错误或未设置。请输入访问令牌：');if(!t2){add('err','已取消：缺少访问令牌');send.disabled=false;send.textContent='发送';return;}localStorage.setItem('btoken',t2);r=await fetch('/v1/agent',{method:'POST',headers:authHeaders(),body:JSON.stringify(payload)});}
    if(!r.ok){add('err','请求失败: HTTP '+r.status);return;}
    const reader=r.body.getReader(),dec=new TextDecoder();let buf='';
    while(true){
      const {done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;
      while((i=buf.indexOf('\\n\\n'))>=0){
        const block=buf.slice(0,i);buf=buf.slice(i+2);
        for(const line of block.split('\\n')){
          if(!line.startsWith('data: '))continue;
          let ev;try{ev=JSON.parse(line.slice(6));}catch(e){continue;}
          if(ev.type==='stage'){if(ev.stage==='ready'){readyTs=Date.now();setStage('就绪 · '+ev.model,'ready');}else if(ev.stage==='load')setStage('加载模型…','ready');else setStage('推理中 · 步骤 '+ev.step,'infer');}
          else if(ev.type==='tool'){setStage('工具执行: '+ev.tool,'tool');addStep(ev);}
          else if(ev.type==='usage'){setCtx(ev.prompt_tokens,ev.completion_tokens);}
          else if(ev.type==='done'){
            const j=ev.result,msg=j.choices&&j.choices[0]&&j.choices[0].message?j.choices[0].message:null;
            const c=msg?msg.content:'';
            const reason=msg?String(msg.reasoning_content||msg.thinking||'').trim():'';
            if(c){
              const d=add('assistant',c);
              if(reason&&reason!==String(c).trim()){const rp=document.createElement('details');rp.className='think';const sg=document.createElement('summary');sg.textContent='思考过程 · '+((Date.now()-(readyTs||t0))/1000).toFixed(1)+'s';const rb=document.createElement('div');rb.className='tb';rb.textContent=reason;rp.appendChild(sg);rp.appendChild(rb);d.insertBefore(rp,d.firstChild);}
              appendMeta(d,j.agent_meta);history.push({role:'assistant',content:c});
            }
            else add('err','无回复: '+(j.error?JSON.stringify(j):JSON.stringify(j).slice(0,200)));
            updateSum(j.agent_meta);setStage('完成','done');
          }
          else if(ev.type==='error'){add('err',ev.message);setStage('失败','err');}
        }
      }
    }
    if(history.length>24)history=history.slice(-24);
  }catch(e){add('err','请求失败: '+e);setStage('失败','err');}
  send.disabled=false;send.textContent='发送';text.focus();
}
function setStage(t,c){document.getElementById('stageDot').className='dot '+c;document.getElementById('stageText').textContent=t;}
function setCtx(p,c){const lim=8192,pct=Math.min(100,Math.round(p/lim*100));document.getElementById('ctxBar').style.width=pct+'%';document.getElementById('ctxText').textContent=p+' / '+lim+' tok';}
function addStep(ev){
  const st=document.getElementById('steps');
  if(st.querySelector('.hint'))st.innerHTML='';
  const d=document.createElement('div');d.className='step';
  d.innerHTML='<span class="snum">'+ev.step+'</span><span class="sname">'+esc(ev.tool)+'</span><span class="stime">'+ev.elapsed+'s</span>'
    +'<div class="sargs">'+esc(JSON.stringify(ev.args))+'</div>'
    +'<div class="sres">'+esc(String(ev.result||'').replace(/\s+/g,' ').slice(0,80))+'</div>';
  st.appendChild(d);st.scrollTop=st.scrollHeight;
}
function resetSide(){
  document.getElementById('steps').innerHTML='<div class="hint">暂无</div>';
  document.getElementById('ctxBar').style.width='0%';
  document.getElementById('ctxText').textContent='0 / 8192 tok';
  document.getElementById('sumText').textContent='—';
  setStage('处理中…','load');
}
function updateSum(m){
  if(!m)return;
  document.getElementById('sumText').innerHTML='模型 '+esc(m.model)+'<br>耗时 '+m.elapsed+'s · '+m.tok_per_sec+' tok/s<br>上下文 '+m.context_tokens+' tok · 输出 '+m.completion_tokens+' tok<br>工具步数 '+m.steps;
}
send.onclick=ask;text.addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
</script>
</body>
</html>"""

    def _serve_chat_page(self):
        html = self._CHAT_HTML.replace("__DEFAULT_MODEL__", str(self.server.gw.config.get("default_model", "")))
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_models(self):
        gw: Gateway = self.server.gw
        # 本地后端优先；不可达则返回索引模型
        local = gw.pick_backend("local")
        if local and local.is_ready():
            status, resp = local.request("GET", "/models", timeout=10)
            if status == 200:
                data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        idx_models = gw.index.list_models()
        data = [{"id": m, "object": "model", "created": int(time.time()), "owned_by": "bionic-lite"} for m in idx_models]
        self._json(200, {"object": "list", "data": data})

    def _handle_chat(self):
        gw: Gateway = self.server.gw
        try:
            body = json.loads(self._read_body())
        except Exception:
            return self._error(400, "Invalid JSON body")
        if body.get("agent"):
            # 客户端在 body 中显式标记 agent=true 时走 Agent Loop
            return self._run_agent(body)
        req_model = body.get("model") or gw.config.get("default_model")
        text = self._msg_text_list(body.get("messages", []))
        target = gw.router.route(req_model, text) or gw.config.get("default_model")
        backend = gw.pick_backend("local")
        if backend is None:
            return self._error(503, "无可用后端，请检查 config.json backends")

        # 热切换：lmstudio 后端驻留模型不符时自动切换
        if isinstance(backend, LmStudioBackend):
            ok, msg = backend.hot_switch(target, gw.config["lms_cli"],
                                         gw.config["load_context"], gw.config["load_gpu"])
            if not ok:
                return self._error(502, f"模型热切换失败: {msg}")

        body["model"] = target
        stream = bool(body.get("stream", False))
        if stream:
            self._proxy_stream(backend, body)
        else:
            self._proxy_json(backend, body)

    def _handle_agent(self):
        """独立端点 POST /v1/agent：Agent Loop 执行。stream=true 时走 SSE 实时事件流。"""
        try:
            body = json.loads(self._read_body())
        except Exception:
            return self._error(400, "Invalid JSON body")
        if body.get("stream"):
            return self._run_agent_stream(body)
        self._run_agent(body)

    def _prepare_agent(self, body):
        """路由+后端解析，返回 (target, backend, err)。"""
        gw: Gateway = self.server.gw
        req_model = body.get("model") or gw.config.get("default_model")
        text = self._msg_text_list(body.get("messages", []))
        target = gw.router.route(req_model, text) or gw.config.get("default_model")
        backend = gw.pick_backend("local")
        if backend is None:
            return None, None, "无可用后端，请检查 config.json backends"
        # 热切换：agent 循环内全程保持同一模型，避免中途丢 KV
        if isinstance(backend, LmStudioBackend):
            ok, msg = backend.hot_switch(target, gw.config["lms_cli"],
                                         gw.config["load_context"], gw.config["load_gpu"])
            if not ok:
                return None, None, f"模型热切换失败: {msg}"
        return target, backend, None

    def _run_agent(self, body):
        target, backend, err = self._prepare_agent(body)
        if err:
            return self._error(502 if "热切换" in err else 503, err)
        loop = AgentLoop(self.server.gw, backend, target, max_steps=body.get("max_steps"))
        extra = {}
        if body.get("chat_template_kwargs"):
            extra["chat_template_kwargs"] = body["chat_template_kwargs"]
        self._json(200, loop.run(body.get("messages", []), extra_kwargs=extra or None))

    def _run_agent_stream(self, body):
        """SSE：按 AgentLoop 事件实时推送 stage/tool/usage，最后推送 done(result)。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def emit(obj):
            try:
                self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            except Exception:
                pass

        emit({"type": "stage", "stage": "load", "msg": "解析路由并加载模型…"})
        target, backend, err = self._prepare_agent(body)
        if err:
            emit({"type": "error", "message": err})
            self.close_connection = True
            return
        emit({"type": "stage", "stage": "ready", "model": target})
        loop = AgentLoop(self.server.gw, backend, target, max_steps=body.get("max_steps"))
        extra = {}
        if body.get("chat_template_kwargs"):
            extra["chat_template_kwargs"] = body["chat_template_kwargs"]
        result = loop.run(body.get("messages", []), extra_kwargs=extra or None, on_event=emit)
        emit({"type": "done", "result": result})
        self.close_connection = True

    def _handle_embedding(self):
        gw: Gateway = self.server.gw
        try:
            body = json.loads(self._read_body())
        except Exception:
            return self._error(400, "Invalid JSON body")
        backend = gw.pick_backend("local")
        if backend is None:
            return self._error(503, "无可用后端")
        self._proxy_json(backend, body)

    # ---------- 转发实现 ----------
    def _proxy_json(self, backend, body):
        try:
            status, resp = backend.request("POST", "/chat/completions", body, timeout=600)
            data = resp.read()
        except urllib.error.HTTPError as e:
            return self._error(502, f"后端错误: {e.code} {e.read().decode(errors='ignore')[:500]}")
        except Exception as e:
            return self._error(502, f"后端不可达: {e}")
        # content 回填（对齐 Bionic 强制思考模式场景）
        if status == 200:
            try:
                obj = json.loads(data)
                for ch in obj.get("choices", []):
                    msg = ch.get("message", {})
                    if not msg.get("content") and msg.get("reasoning_content"):
                        msg["content"] = msg["reasoning_content"]
                data = json.dumps(obj, ensure_ascii=False).encode()
            except Exception:
                pass
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy_stream(self, backend, body):
        try:
            status, resp = backend.request("POST", "/chat/completions", body, timeout=600)
        except Exception as e:
            return self._error(502, f"后端不可达: {e}")
        if status != 200:
            try:
                detail = resp.read().decode(errors="ignore").strip()[:500]
            except Exception:
                detail = ""
            return self._error(status if status < 600 else 502,
                               f"后端错误 {status}: {detail or '(无响应体)'}")
        self.send_response(status if status else 200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                # 逐 SSE 事件改写：delta.content 为空时用 reasoning_content 回填
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.startswith(b"data: "):
                        payload = line[6:]
                        if payload.strip() == b"[DONE]":
                            self.wfile.write(line + b"\n")
                            continue
                        try:
                            obj = json.loads(payload)
                            for ch in obj.get("choices", []):
                                dl = ch.get("delta", {})
                                if not dl.get("content") and dl.get("reasoning_content"):
                                    dl["content"] = dl["reasoning_content"]
                            line = b"data: " + json.dumps(obj, ensure_ascii=False).encode()
                        except Exception:
                            pass
                    self.wfile.write(line + b"\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                resp.close()
            except Exception:
                pass


class Gateway:
    """网关主体：配置 / 后端 / 路由 / 索引 的组合。"""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.index = ModelIndex(self.config.get("models_dir", ""))
        self.router = Router(self.config)
        self.backends = self._build_backends()
        self.tools = ToolRegistry(self)

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            merged = dict(DEFAULT_CONFIG)
            merged.update(user_cfg)
            return merged
        return dict(DEFAULT_CONFIG)

    def _build_backends(self):
        b = []
        for cfg in self.config.get("backends", []):
            if cfg.get("type") == "lmstudio":
                b.append(LmStudioBackend(cfg))
            else:
                b.append(Backend(cfg))
        return b

    def pick_backend(self, name=None):
        if name:
            for bk in self.backends:
                if bk.cfg.get("name") == name:
                    return bk
        return self.backends[0] if self.backends else None

    def dump_config_template(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)

    def start(self, auto_open=False):
        self.index.scan()
        srv = ThreadingHTTPServer((self.config["host"], self.config["port"]), GatewayHandler)
        srv.gw = self
        print(f"[bionic-lite] 网关启动: http://{self.config['host']}:{self.config['port']}/v1")
        print(f"[bionic-lite] 网页对话页: http://{self.config['host']}:{self.config['port']}/")
        print(f"[bionic-lite] 索引模型数: {len(self.index.index)}")
        print(f"[bionic-lite] 后端: {[b.cfg['name'] for b in self.backends]}")
        print(f"[bionic-lite] 路由规则: {len(self.router.rules)} 条")
        if auto_open:
            import webbrowser
            webbrowser.open(f"http://{self.config['host']}:{self.config['port']}/")
        srv.serve_forever()


def cmd_chat(gw):
    """CLI 交互对话：后台起网关线程，主线程 REPL 调用 /v1/agent。"""
    import urllib.request

    srv = ThreadingHTTPServer((gw.config["host"], gw.config["port"]), GatewayHandler)
    srv.gw = gw
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://{gw.config['host']}:{gw.config['port']}/v1"
    print(f"[bionic-lite] 网关线程已启动: {base}")
    print("[bionic-lite] 输入问题开始对话（agent 工具自动可用），/exit 退出，/think on|off 开关思考模式（默认 on）")
    history = []
    think = True
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit", "q"):
            break
        if line in ("/think on", "/think off"):
            think = line.endswith("on")
            print(f"[bionic-lite] 思考模式已{'开启' if think else '关闭'}")
            continue
        history.append({"role": "user", "content": line})
        if len(history) > 24:
            history = history[-24:]
        payload = {"messages": history}
        if not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            base + "/agent",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            c = j["choices"][0]["message"]["content"]
            print("AI>", c)
            history.append({"role": "assistant", "content": c})
        except Exception as e:
            print(f"[bionic-lite] 请求失败: {e}")
    print("[bionic-lite] 再见")


def main():
    ap = argparse.ArgumentParser(description="Bionic Lite — 本地 LLM 多后端调度网关")
    ap.add_argument("cmd", choices=["serve", "models", "status", "init", "chat", "web"],
                    help="serve=启动网关; models=扫描索引; status=检查后端; init=生成配置模板; chat=CLI 对话; web=启动网关并打开网页对话页")
    ap.add_argument("--config", default="config.json", help="配置文件路径")
    ap.add_argument("--port", type=int, default=None, help="覆盖监听端口")
    args = ap.parse_args()

    gw = Gateway(args.config)
    if args.port:
        gw.config["port"] = args.port

    if args.cmd == "init":
        if Path(args.config).exists():
            print(f"[bionic-lite] {args.config} 已存在，跳过")
        else:
            gw.dump_config_template(args.config)
            print(f"[bionic-lite] 已生成配置模板: {args.config}")
        return

    if args.cmd == "models":
        idx = gw.index.scan()
        print(f"[bionic-lite] 索引模型 {len(idx)} 个:")
        for m in gw.index.list_models():
            print(f"  {m}")
        return

    if args.cmd == "status":
        for bk in gw.backends:
            ok = bk.is_ready()
            print(f"[bionic-lite] 后端 {bk.cfg['name']} ({bk.cfg['type']}): {'就绪' if ok else '不可达'}")
        return

    if args.cmd == "chat":
        cmd_chat(gw)
        return

    if args.cmd == "web":
        gw.start(auto_open=True)
        return

    gw.start()


if __name__ == "__main__":
    main()
