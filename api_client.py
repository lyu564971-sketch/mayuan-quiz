"""API 客户端模块。

封装与 LLM API 的通信，包含重试、退避和速率限制逻辑。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import openai

from config import ModelConfig


@dataclass
class LLMResponse:
    """LLM API 调用结果。

    Attributes:
        raw_text: 模型返回的原始文本。
        model: 使用的模型名称。
        usage_tokens: token 使用统计（prompt_tokens + completion_tokens）。
        latency_seconds: API 调用耗时秒数。
    """

    raw_text: str
    model: str
    usage_tokens: int = 0
    latency_seconds: float = 0.0


class LLMGenerationError(Exception):
    """LLM 生成失败的自定义异常。

    Attributes:
        message: 错误描述。
        attempts: 已尝试的重试次数。
        last_error: 最后一次捕获的原始异常。
    """

    def __init__(self, message: str, attempts: int = 0, last_error: Exception = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class APIClient:
    """LLM API 客户端。

    基于 openai.OpenAI 实现，支持智谱和 DeepSeek 等兼容 OpenAI 接口的服务。

    重试策略：
        - RateLimitError (429): 指数退避 5s × 2^n
        - APIConnectionError / APITimeoutError: 退避 2s × 2^n
        - InternalServerError (5xx): 退避 2s × 2^n
        - 最多重试 max_retries 次
    """

    def __init__(self, config: ModelConfig, max_retries: int = 3):
        """初始化 API 客户端。

        Args:
            config: 模型配置（来自 AppConfig.model）。
            max_retries: 最大重试次数。
        """
        self.config = config
        self.max_retries = max_retries
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=120.0,  # 默认超时，单次调用由 call() 中的 request_timeout 覆盖
        )

        # 统计信息
        self.total_calls: int = 0
        self.total_tokens: int = 0
        self.total_retries: int = 0

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        request_timeout: int = 60,
    ) -> LLMResponse:
        """调用 LLM API 发送消息并获取回复。

        包含完整的重试逻辑和指数退避。

        Args:
            messages: 消息列表，格式 [{"role": "system"|"user"|"assistant", "content": "..."}]。
            temperature: 生成温度。
            max_tokens: 最大输出 token 数。
            request_timeout: 请求超时秒数。

        Returns:
            LLMResponse 实例。

        Raises:
            LLMGenerationError: 所有重试耗尽后。
        """
        last_error: Exception = None
        attempt: int = 0

        for attempt in range(self.max_retries + 1):
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout,
                )
                latency = time.time() - start_time

                choice = response.choices[0]
                content = choice.message.content or ""

                usage = response.usage
                usage_tokens = (
                    (usage.prompt_tokens + usage.completion_tokens)
                    if usage
                    else 0
                )

                self.total_calls += 1
                self.total_tokens += usage_tokens

                # 遵守速率限制
                if self.config.request_interval > 0:
                    time.sleep(self.config.request_interval)

                return LLMResponse(
                    raw_text=content,
                    model=self.config.model_name,
                    usage_tokens=usage_tokens,
                    latency_seconds=latency,
                )

            except openai.RateLimitError as e:
                last_error = e
                self.total_retries += 1
                if attempt < self.max_retries:
                    wait = 5.0 * (2 ** attempt)
                    time.sleep(wait)
                continue

            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last_error = e
                self.total_retries += 1
                if attempt < self.max_retries:
                    wait = 2.0 * (2 ** attempt)
                    time.sleep(wait)
                continue

            except openai.InternalServerError as e:
                last_error = e
                self.total_retries += 1
                if attempt < self.max_retries:
                    wait = 2.0 * (2 ** attempt)
                    time.sleep(wait)
                continue

        raise LLMGenerationError(
            message=(
                f"LLM API 调用失败，已重试 {self.max_retries} 次。"
                f"最后错误: {type(last_error).__name__}: {last_error}"
            ),
            attempts=attempt,
            last_error=last_error,
        )

    def stream_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 800,
        request_timeout: int = 30,
    ):
        """流式调用 LLM API，逐块返回文本。

        Yields:
            每次返回一个文本片段（str）。

        Raises:
            LLMGenerationError: 所有重试耗尽后。
        """
        last_error: Exception = None
        attempt: int = 0

        for attempt in range(self.max_retries + 1):
            try:
                stream = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

                self.total_calls += 1
                return  # 正常结束

            except openai.RateLimitError as e:
                last_error = e
                self.total_retries += 1
                if attempt < self.max_retries:
                    time.sleep(5.0 * (2 ** attempt))
                continue

            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last_error = e
                self.total_retries += 1
                if attempt < self.max_retries:
                    time.sleep(2.0 * (2 ** attempt))
                continue

            except openai.InternalServerError as e:
                last_error = e
                self.total_retries += 1
                if attempt < self.max_retries:
                    time.sleep(2.0 * (2 ** attempt))
                continue

        raise LLMGenerationError(
            message=(
                f"LLM 流式调用失败，已重试 {self.max_retries} 次。"
                f"最后错误: {type(last_error).__name__}: {last_error}"
            ),
            attempts=attempt,
            last_error=last_error,
        )
