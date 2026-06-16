"""LLM 批量生成马原题库解析 —— 主入口 CLI。

通过 argparse 接收命令行参数，协调 LLM API 调用、断点续传、
结果合并等模块完成全量或部分题目的解析生成。
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from config import AppConfig, ModelConfig
from api_client import APIClient, LLMGenerationError, LLMResponse
from validator import OutputValidator, ValidationResult
from checkpoint import CheckpointManager, Checkpoint
from prompts import PromptManager
from result_merger import merge_results, backup_original, build_summary


def load_questions(path: str, sample: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载题目 JSON 文件。

    Args:
        path: 文件路径。
        sample: 如果指定，只取前 N 题。

    Returns:
        题目列表。
    """
    if not os.path.exists(path):
        print(f"错误: 找不到输入文件 '{path}'，请确保 questions.json 存在。", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    if sample and sample > 0:
        questions = questions[:sample]
        print(f"采样模式: 仅处理前 {len(questions)} 题")

    return questions


def parse_llm_json(text: str) -> Optional[Dict[str, str]]:
    """尝试从 LLM 返回文本中解析 JSON。

    支持：
    1. 纯 JSON 对象。
    2. Markdown 代码块包裹的 JSON（```json ... ```）。
    3. 正则提取字段作为兜底。

    Args:
        text: LLM 返回的原始文本。

    Returns:
        解析出的字段字典，包含 location/solution/options_analysis/warning。
        如果所有解析方式均失败，返回 None。
    """
    import re

    # 方式1：纯 JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return {
                "location": result.get("location", ""),
                "solution": result.get("solution", ""),
                "options_analysis": result.get("options_analysis", ""),
                "warning": result.get("warning", ""),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # 方式2：提取 Markdown 代码块中的 JSON
    code_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_matches = re.findall(code_pattern, text)
    for match in code_matches:
        try:
            result = json.loads(match.strip())
            if isinstance(result, dict):
                return {
                    "location": result.get("location", ""),
                    "solution": result.get("solution", ""),
                    "options_analysis": result.get("options_analysis", ""),
                    "warning": result.get("warning", ""),
                }
        except (json.JSONDecodeError, TypeError):
            continue

    # 方式3：正则提取四段内容（兜底）
    location_match = re.search(
        r"(?:location|考点定位)[：:\s]*[\"']?\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE
    )
    solution_match = re.search(
        r"(?:solution|解题思路)[：:\s]*[\"']?\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE
    )
    options_match = re.search(
        r"(?:options_analysis|选项分析)[：:\s]*[\"']?\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE
    )
    warning_match = re.search(
        r"(?:warning|易错提醒)[：:\s]*[\"']?\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE
    )

    # 如果至少提取到两个字段，视为成功
    extracted = {
        "location": location_match.group(1) if location_match else "",
        "solution": solution_match.group(1) if solution_match else "",
        "options_analysis": options_match.group(1) if options_match else "",
        "warning": warning_match.group(1) if warning_match else "",
    }

    non_empty_count = sum(1 for v in extracted.values() if v)
    if non_empty_count >= 2:
        return extracted

    return None


def generate_one(
    question: Dict[str, Any],
    client: APIClient,
    prompt_manager: PromptManager,
    validator: OutputValidator,
    model_config: ModelConfig,
    max_retries: int,
    request_timeout: int = 60,
) -> tuple[Optional[str], int]:
    """为单道题目生成解析。

    包含完整的重试逻辑（temperature 递增）、JSON 解析、降级兜底。

    Args:
        question: 题目字典。
        client: API 客户端。
        prompt_manager: Prompt 管理器。
        validator: 输出校验器。
        model_config: 模型配置。
        max_retries: 最大重试次数。

    Returns:
        (explanation_text, estimated_tokens) — 解析文本和估算 token 消耗。
        解析文本为 None 表示生成失败。
    """
    messages = prompt_manager.build_messages(question)
    temperature = model_config.temperature
    estimated_tokens = 0

    for attempt in range(max_retries + 1):
        try:
            response: LLMResponse = client.call(
                messages=messages,
                temperature=min(temperature, 0.6),
                max_tokens=model_config.max_tokens,
                request_timeout=request_timeout,
            )
            estimated_tokens += response.usage_tokens
            raw_text = response.raw_text

            # 尝试 JSON 解析
            parsed = parse_llm_json(raw_text)

            if parsed is not None:
                # JSON 解析成功，组装四段式
                formatted = validator.format_from_json(parsed)
                # 校验
                validation = validator.validate(formatted)
                if validation.is_valid:
                    return formatted, estimated_tokens
                else:
                    # 校验失败，继续尝试
                    pass
            else:
                # JSON 解析失败，尝试直接校验原始文本
                validation = validator.validate(raw_text)
                if validation.is_valid:
                    return raw_text, estimated_tokens

        except LLMGenerationError:
            # 重试耗尽
            break

        # 递增 temperature 再试
        temperature = min(temperature + 0.05, 0.6)

    # 所有重试用尽，使用降级模板
    try:
        fallback = prompt_manager.build_fallback_explanation(question)
        return fallback, estimated_tokens
    except Exception:
        return None, estimated_tokens


def run_generation(
    questions: List[Dict[str, Any]],
    config: AppConfig,
    force: bool = False,
    in_place: bool = False,
) -> int:
    """执行批量解析生成的主循环。

    Args:
        questions: 题目列表。
        config: 应用配置。
        force: 是否忽略 checkpoint 重新处理。
        in_place: 是否直接覆盖原文件。

    Returns:
        退出码 (0=成功, 1=部分失败)。
    """
    # 初始化组件
    client = APIClient(config=config.model, max_retries=config.max_retries)
    prompt_manager = PromptManager()
    validator = OutputValidator(min_total_length=config.min_explanation_length)
    checkpoint_mgr = CheckpointManager(checkpoint_path=config.checkpoint_file)

    # 加载或创建 checkpoint
    checkpoint = None if force else checkpoint_mgr.load()

    if checkpoint is not None:
        print(f"检测到断点文件: {config.checkpoint_file}")
        print(f"  已完成: {len(checkpoint.completed_ids)} 题")
        print(f"  已失败: {len(checkpoint.failed_ids)} 题")
        print(f"  模型: {checkpoint.model_used}")

        if not force and len(checkpoint.completed_ids) >= len(questions):
            print("所有题目已处理完毕。使用 --force 可重新处理。")
            return 0
    else:
        checkpoint = checkpoint_mgr.init_checkpoint(
            total_questions=len(questions),
            model_used=config.model.model_name,
        )
        checkpoint_mgr.save(checkpoint, config.checkpoint_file)

    # 获取待处理题目
    pending = checkpoint_mgr.get_pending_indices(questions)
    if not pending:
        print("所有题目已处理完毕，无待处理题目。")
        return 0

    print(f"\n开始处理 {len(pending)} 道题目（模型: {config.model.model_name}）")
    print(f"温度: {config.model.temperature} | 最大 tokens: {config.model.max_tokens}")
    print(f"重试次数: {config.max_retries} | 速率间隔: {config.model.request_interval}s")
    print()

    # 累计结果
    explanations: Dict[int, str] = {}
    # 累加已有结果
    for q_id in checkpoint.completed_ids:
        # 从题目中加载已有解析（如果有 llm_explanation 字段）
        for q in questions:
            if q.get("id") == q_id:
                existing = q.get("llm_explanation", "")
                if existing:
                    explanations[q_id] = existing
                break

    total_tokens = 0
    start_time = time.time()

    # 主循环（带 tqdm 进度条）
    with tqdm(total=len(questions), initial=checkpoint.current_index,
              desc="生成解析", unit="题") as pbar:
        for idx, question in pending:
            q_id = question.get("id", idx + 1)
            q_type = question.get("type", "single")
            q_type_cn = {"single": "单选", "multiple": "多选", "judge": "判断"}.get(
                q_type, q_type
            )
            pbar.set_postfix_str(f"ID={q_id} {q_type_cn}")

            explanation, est_tokens = generate_one(
                question=question,
                client=client,
                prompt_manager=prompt_manager,
                validator=validator,
                model_config=config.model,
                max_retries=config.max_retries,
                request_timeout=config.request_timeout,
            )

            total_tokens += est_tokens

            if explanation is not None:
                explanations[q_id] = explanation
                checkpoint_mgr.mark_completed(q_id)
            else:
                # 使用降级模板兜底
                fallback = prompt_manager.build_fallback_explanation(question)
                explanations[q_id] = fallback
                checkpoint_mgr.mark_failed(q_id)

            # 每处理完一题，更新 checkpoint
            checkpoint_mgr.save(checkpoint_mgr.checkpoint, config.checkpoint_file)
            pbar.update(1)

    elapsed = time.time() - start_time

    # 合并结果
    input_path = config.input_file if in_place else None
    output_path = merge_results(
        questions=questions,
        explanations=explanations,
        output_path=config.output_file,
        in_place=in_place,
        input_path=input_path,
    )

    # 打印摘要
    failed_ids = list(checkpoint_mgr.checkpoint.failed_ids if checkpoint_mgr.checkpoint else [])
    summary = build_summary(
        questions=questions,
        explanations={
            k: v for k, v in explanations.items()
            if k not in (checkpoint_mgr.checkpoint.failed_ids if checkpoint_mgr.checkpoint else set())
        },
        total_time=elapsed,
        estimated_tokens=total_tokens,
        failed_ids=failed_ids,
    )

    print(f"\n{'=' * 60}")
    print(f"  处理完成")
    print(f"{'=' * 60}")
    print(f"  总题目数:      {summary['total_questions']}")
    print(f"  成功:          {summary['succeeded']}")
    print(f"  失败:          {summary['failed']}")
    print(f"  成功率:        {summary['success_rate']}")
    print(f"  总耗时:        {summary['total_time_seconds']:.1f} 秒")
    print(f"  平均每题:      {summary['avg_time_per_question']:.2f} 秒")
    print(f"  估算 Token:    {summary['estimated_token_consumption']}")
    print(f"  输出文件:      {output_path}")
    if summary["failed_ids"]:
        print(f"  失败题目 ID:   {summary['failed_ids']}")
    print(f"{'=' * 60}")

    return 0 if summary["failed"] == 0 else 1


def main() -> None:
    """CLI 主入口。"""
    parser = argparse.ArgumentParser(
        description="马原题库 LLM 批量解析生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用智谱（默认）处理全部题目
  python run_llm.py

  # 使用 DeepSeek
  python run_llm.py --model deepseek

  # 忽略断点重新开始
  python run_llm.py --force

  # 仅处理前 10 题
  python run_llm.py --sample 10

  # 直接覆盖原文件
  python run_llm.py --in-place

环境变量:
  ZHIPU_API_KEY                    智谱 AI API 密钥
  DEEPSEEK_API_KEY                 DeepSeek API 密钥
  ACTIVE_PROVIDER                  当前提供商 (zhipu/deepseek), 默认 zhipu
  TEMPERATURE                      生成温度, 默认 0.3
  MAX_TOKENS                       最大输出 tokens, 默认 2000
        """,
    )
    parser.add_argument(
        "--model",
        choices=["zhipu", "deepseek"],
        default=None,
        help="LLM 提供商 (默认: 由 ACTIVE_PROVIDER 环境变量决定)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有断点，重新处理全部",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="直接覆盖 questions.json（会先备份为 .bak）",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="仅处理前 N 题",
    )

    args = parser.parse_args()

    # 如果通过命令行指定了 model，覆盖环境变量
    if args.model:
        os.environ["ACTIVE_PROVIDER"] = args.model

    # 加载配置
    try:
        config = AppConfig.load()
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 加载题目
    questions = load_questions(config.input_file, sample=args.sample)

    if not questions:
        print("题目列表为空，退出。", file=sys.stderr)
        sys.exit(1)

    print(f"已加载 {len(questions)} 道题目")
    print(f"当前模型: {config.model.model_name} ({config.active_provider})")

    # 执行生成
    exit_code = run_generation(
        questions=questions,
        config=config,
        force=args.force,
        in_place=args.in_place,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
