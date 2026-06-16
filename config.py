"""应用配置模块。

定义 ModelConfig 和 AppConfig 两个 dataclass，负责从环境变量加载 LLM 集成所需的所有配置。
"""

from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class ModelConfig:
    """单个 LLM 模型的配置。

    Attributes:
        model_name: 模型名称标识符。
        base_url: API 端点基础 URL。
        api_key: API 密钥（从环境变量读取）。
        temperature: 生成温度，控制输出随机性。
        max_tokens: 最大输出 token 数。
        request_interval: 请求间隔秒数，用于遵守速率限制。
    """

    model_name: str
    base_url: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 2000
    request_interval: float = 0.0


# 预置模型配置
PRESET_ZHIPU = {
    "model_name": "glm-4-flash",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    "request_interval": 0.0,
}

PRESET_DEEPSEEK = {
    "model_name": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "request_interval": 30.0,
}


def _require_env(var_name: str, description: str) -> str:
    """读取必需的环境变量，未设置时抛出清晰的错误提示。

    Args:
        var_name: 环境变量名。
        description: 该变量的中文描述。

    Returns:
        环境变量的值。

    Raises:
        ValueError: 当环境变量未设置时。
    """
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise ValueError(
            f"环境变量 {var_name} 未设置。"
            f"请设置 {var_name} 为您的 {description}。\n"
            f"示例：export {var_name}=your_key_here"
        )
    return value


@dataclass
class AppConfig:
    """应用全局配置。

    通过 AppConfig.load() 类方法从 os.environ 加载所有配置。
    未提供 ACTIVE_PROVIDER 时默认使用智谱（zhipu）。

    Attributes:
        active_provider: 当前使用的 LLM 提供商名称（"zhipu" 或 "deepseek"）。
        model: 当前激活的 ModelConfig 实例。
        input_file: 输入题库 JSON 文件路径。
        output_file: 输出结果 JSON 文件路径。
        checkpoint_file: 断点续传文件路径。
        max_retries: API 调用最大重试次数。
        min_explanation_length: 解析文本最小长度要求。
        request_timeout: API 请求超时秒数。
    """

    active_provider: str
    model: ModelConfig
    input_file: str = "questions.json"
    output_file: str = "questions_with_llm.json"
    checkpoint_file: str = "checkpoint.json"
    max_retries: int = 3
    min_explanation_length: int = 200
    request_timeout: int = 60

    @classmethod
    def load(cls) -> "AppConfig":
        """从环境变量加载完整应用配置。

        Returns:
            配置好的 AppConfig 实例。

        Raises:
            ValueError: 当必需的 API 密钥未设置时。
        """
        provider = os.environ.get("ACTIVE_PROVIDER", "zhipu").strip().lower()

        temperature = float(os.environ.get("TEMPERATURE", "0.3"))
        max_tokens = int(os.environ.get("MAX_TOKENS", "2000"))

        if provider == "zhipu":
            api_key = _require_env("ZHIPU_API_KEY", "智谱AI API密钥")
            model = ModelConfig(
                model_name=PRESET_ZHIPU["model_name"],
                base_url=PRESET_ZHIPU["base_url"],
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                request_interval=PRESET_ZHIPU["request_interval"],
            )
        elif provider == "deepseek":
            api_key = _require_env("DEEPSEEK_API_KEY", "DeepSeek API密钥")
            model = ModelConfig(
                model_name=PRESET_DEEPSEEK["model_name"],
                base_url=PRESET_DEEPSEEK["base_url"],
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                request_interval=PRESET_DEEPSEEK["request_interval"],
            )
        else:
            raise ValueError(
                f"未知的 LLM 提供商: {provider}。"
                f"请将 ACTIVE_PROVIDER 设置为 'zhipu' 或 'deepseek'。"
            )

        return cls(
            active_provider=provider,
            model=model,
            input_file=os.environ.get("LLM_INPUT_FILE", "questions.json"),
            output_file=os.environ.get("LLM_OUTPUT_FILE", "questions_with_llm.json"),
            checkpoint_file=os.environ.get(
                "LLM_CHECKPOINT_FILE", "checkpoint.json"
            ),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            min_explanation_length=int(
                os.environ.get("LLM_MIN_EXPLANATION_LENGTH", "200")
            ),
            request_timeout=int(os.environ.get("LLM_REQUEST_TIMEOUT", "60")),
        )


