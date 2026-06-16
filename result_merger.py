"""结果合并模块。

将 LLM 生成的解析结果合并到题目 JSON 中并写出最终文件。
"""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


def backup_original(file_path: str) -> str:
    """备份原文件为 .bak 后缀。

    Args:
        file_path: 原文件路径。

    Returns:
        备份文件路径。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在，无法备份: {file_path}")

    bak_path = file_path + ".bak"
    shutil.copy2(file_path, bak_path)
    return bak_path


def merge_results(
    questions: List[Dict[str, Any]],
    explanations: Dict[int, str],
    output_path: str,
    in_place: bool = False,
    input_path: Optional[str] = None,
) -> str:
    """将 LLM 解析结果合并到题目列表并写出 JSON 文件。

    Args:
        questions: 原始题目列表。
        explanations: 题目 ID 到解析文本的映射。
        output_path: 输出文件路径。
        in_place: 是否覆盖原文件（会先备份）。
        input_path: in_place 模式下的原文件路径。

    Returns:
        实际输出文件路径。
    """
    # 深拷贝题目列表，避免修改原数据
    merged: List[Dict[str, Any]] = []
    merged_count = 0

    for q in questions:
        q_copy = dict(q)
        q_id = q_copy.get("id")

        if q_id is not None and q_id in explanations:
            q_copy["llm_explanation"] = explanations[q_id]
            merged_count += 1
        else:
            q_copy["llm_explanation"] = ""

        merged.append(q_copy)

    # 如果指定 in-place 且提供了 input_path，先备份
    if in_place and input_path:
        backup_original(input_path)
        actual_output = input_path
    else:
        actual_output = output_path

    # 写入结果
    with open(actual_output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return actual_output


def build_summary(
    questions: List[Dict[str, Any]],
    explanations: Dict[int, str],
    total_time: float,
    estimated_tokens: int,
    failed_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """构建处理摘要。

    Args:
        questions: 原始题目列表。
        explanations: 成功的解析映射。
        total_time: 总耗时秒数。
        estimated_tokens: 估算 token 消耗。
        failed_ids: 失败题目 ID 列表。

    Returns:
        摘要信息字典。
    """
    failed_ids = failed_ids or []
    total = len(questions)
    succeeded = len(explanations)
    failed = len(failed_ids)

    return {
        "total_questions": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": f"{succeeded / total * 100:.1f}%" if total > 0 else "0%",
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_question": (
            round(total_time / succeeded, 2) if succeeded > 0 else 0
        ),
        "estimated_token_consumption": estimated_tokens,
        "failed_ids": sorted(failed_ids),
        "generated_at": datetime.now().isoformat(),
    }
