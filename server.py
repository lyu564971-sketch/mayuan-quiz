"""马原答题助手 - AI 解析代理服务器。Flask + CORS，接收前端请求，调用 GLM-4-Flash 生成解析，返回 JSON。端口 8765"""

import json
import re
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from config import AppConfig
from api_client import APIClient
from prompts import PromptManager

# 默认加载 .env，启动时自动读取 API Key
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).with_name(".env"), override=False)
except Exception:
    pass

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return app.send_static_file("index.html")


# === 初始化（模块级，服务启动时执行一次）===
try:
    app_config = AppConfig.load()
    model_config = app_config.model
    prompt_manager = PromptManager()
    api_client = APIClient(model_config, max_retries=app_config.max_retries)
except Exception as e:
    print(f"[server] 初始化失败: {e}")
    api_client = None
    prompt_manager = None


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if api_client else "error",
        "model": "glm-4-flash",
    })


def _parse_llm_response(text: str) -> dict:
    """解析 LLM 返回内容，支持 JSON 和【】标记两种格式。"""
    import json as _json
    import re as _re

    KEY_MAP = {
        "考点定位": "location", "解题思路": "solution",
        "选项分析": "options_analysis", "易错提醒": "warning",
    }

    def _clean_key(key: str) -> str:
        return _re.sub(r"^[\\s【】\"\'“”]+|[\\s【】\"\'“”]+$", "", str(key)).strip()

    def _normalize(d: dict) -> dict:
        mapped = {}
        for k, v in d.items():
            cleaned = _clean_key(k)
            ek = KEY_MAP.get(cleaned)
            if not ek:
                for zh_key, eng_key in KEY_MAP.items():
                    if zh_key in cleaned:
                        ek = eng_key
                        break
            mapped[ek or cleaned] = v
        return mapped

    try:
        result = _json.loads(text)
        if isinstance(result, dict):
            result = _normalize(result)
            if any(result.get(v) for v in KEY_MAP.values()):
                return {k: v for k, v in result.items()}
    except (_json.JSONDecodeError, TypeError):
        pass

    code_match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_match:
        try:
            result = _json.loads(code_match.group(1).strip())
            if isinstance(result, dict):
                result = _normalize(result)
                if any(result.get(v) for v in KEY_MAP.values()):
                    return {k: v for k, v in result.items()}
        except (_json.JSONDecodeError, TypeError):
            pass

    sections = {}
    for key, title in [("location", "考点定位"), ("solution", "解题思路"),
                        ("options_analysis", "选项分析"), ("warning", "易错提醒")]:
        pattern = rf"【\s*[\"\'“”]?{title}[\"\'“”]?\s*】\s*([\s\S]*?)(?=【|$)"
        m = _re.search(pattern, text)
        sections[key] = m.group(1).strip() if m else ""

    non_empty = sum(1 for v in sections.values() if v)
    if non_empty >= 2:
        return sections

    return {"location": "", "solution": text, "options_analysis": "", "warning": ""}


# 常见的英文 key 模式，需要被剥离
_EN_KEY_PATTERNS = re.compile(
    r"^(step\d+|option[_]?\w*|correct|incorrect|confusion|tips|error[_]?thought"
    r"|analysis|explanation|reason|answer|wrong|right|key|point|note|detail"
    r"|A[：:]|B[：:]|C[：:]|D[：:]|[A-D]项?)\s*[：:]\s*",
    re.IGNORECASE,
)


def _flatten_value(v) -> str:
    """将任意值转为可读字符串。剥离英文key，只保留中文内容。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return _clean_text(v)
    if isinstance(v, dict):
        parts = []
        for kk, vv in v.items():
            val = _flatten_value(vv)
            if not val:
                continue
            # 如果 key 是纯英文/数字标识（如 step1, A, correct），直接用值
            if re.match(r'^[A-Za-z_]\w*$', kk):
                parts.append(val)
            else:
                parts.append(f"{kk}：{val}")
        return "\n".join(parts)
    if isinstance(v, list):
        return "\n".join(_flatten_value(item) for item in v)
    return str(v)


def _clean_text(text: str) -> str:
    """清洗文本：剥离行首的英文 key 标记。"""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # 剥离行首英文key，如 "step1：审题：xxx" → "审题：xxx"
        line = _EN_KEY_PATTERNS.sub("", line.strip())
        # 剥离 "correct：" / "incorrect：" 前缀
        line = re.sub(r"^(correct|incorrect|right|wrong)[：:]\s*", "", line, flags=re.IGNORECASE)
        # 剥离 "confusion：" / "tips：" / "error_thought：" 前缀
        line = re.sub(r"^(confusion|tips|error_thought|note|key_point)[：:]\s*", "", line, flags=re.IGNORECASE)
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def _ensure_str_values(d: dict) -> dict:
    """确保字典所有值都是可读字符串，防止前端 [object Object]。"""
    return {k: _flatten_value(v) for k, v in d.items()}


@app.route("/api/explain", methods=["POST"])
def explain():
    """流式 SSE 端点 - AI 解析。前端通过 EventSource 或 fetch 读取流式数据。"""
    if api_client is None:
        return jsonify({"error": "API 客户端未初始化，请检查 .env 中的 ZHIPU_API_KEY"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "请求体为空或非 JSON"}), 400

    required = ["id", "type", "chapter", "stem", "options", "answer"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"缺少字段: {field}"}), 400

    def generate():
        full_text = ""
        try:
            messages = prompt_manager.build_messages(data)
            # 流式调用，逐步返回文本片段
            for chunk in api_client.stream_call(messages, temperature=0.3, max_tokens=1500, request_timeout=120):
                full_text += chunk
                # SSE 格式：data: {json}\n\n
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"

            # 流结束，解析完整响应并返回结构化结果
            result = _parse_llm_response(full_text)
            result = _ensure_str_values(result)
            final = {
                "type": "done",
                "id": data["id"],
                "location": result.get("location", ""),
                "solution": result.get("solution", ""),
                "options_analysis": result.get("options_analysis", ""),
                "warning": result.get("warning", ""),
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)}, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=port, debug=False)


