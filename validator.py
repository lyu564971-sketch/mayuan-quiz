"""输出校验器模块。

对 LLM 生成的解析文本进行四段式结构校验。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# 四段式必需段落标题
REQUIRED_SECTIONS = ["【考点定位】", "【解题思路】", "【选项分析】", "【易错提醒】"]

# JSON 字段到段落标题的映射
JSON_FIELD_TO_SECTION = {
    "location": "【考点定位】",
    "solution": "【解题思路】",
    "options_analysis": "【选项分析】",
    "warning": "【易错提醒】",
}


@dataclass
class ValidationResult:
    """校验结果。

    Attributes:
        is_valid: 是否通过校验。
        missing_sections: 缺失的段落标题列表。
        total_length: 文本总长度。
        error_message: 校验失败时的错误描述。
    """

    is_valid: bool
    missing_sections: List[str] = field(default_factory=list)
    total_length: int = 0
    error_message: str = ""


class OutputValidator:
    """LLM 输出校验器。

    检查四段式结构完整性、段落内容非空及总长度达标。
    """

    def __init__(
        self,
        required_sections: Optional[List[str]] = None,
        min_section_length: int = 20,
        min_total_length: int = 200,
    ):
        """初始化校验器。

        Args:
            required_sections: 必需段落标题列表，默认 REQUIRED_SECTIONS。
            min_section_length: 每段有效内容的最低字数。
            min_total_length: 整体解析的最低总字数。
        """
        self.required_sections = required_sections or REQUIRED_SECTIONS
        self.min_section_length = min_section_length
        self.min_total_length = min_total_length

    def validate(self, text: str) -> ValidationResult:
        """校验解析文本是否满足四段式结构要求。

        检查规则：
        1. 四个段落标题全部存在。
        2. 每段有效内容 ≥ min_section_length 字符（去除标题和空白）。
        3. 总长度 ≥ min_total_length。

        Args:
            text: LLM 生成的解析文本。

        Returns:
            ValidationResult 实例。
        """
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                missing_sections=list(self.required_sections),
                total_length=0,
                error_message="解析文本为空",
            )

        total_length = len(text)
        missing_sections: List[str] = []
        sections = self.extract_sections(text)

        for section_title in self.required_sections:
            if section_title not in sections:
                missing_sections.append(section_title)
                continue
            content = sections[section_title]
            # 去除标题本身，计算有效内容长度
            effective = content.replace(section_title, "", 1).strip()
            if len(effective) < self.min_section_length:
                missing_sections.append(section_title)
                continue

        if missing_sections:
            return ValidationResult(
                is_valid=False,
                missing_sections=missing_sections,
                total_length=total_length,
                error_message=f"缺失或内容不足的段落: {', '.join(missing_sections)}",
            )

        if total_length < self.min_total_length:
            return ValidationResult(
                is_valid=False,
                missing_sections=[],
                total_length=total_length,
                error_message=(
                    f"总长度 {total_length} 字，低于最低要求 {self.min_total_length} 字"
                ),
            )

        return ValidationResult(
            is_valid=True,
            missing_sections=[],
            total_length=total_length,
            error_message="",
        )

    def extract_sections(self, text: str) -> Dict[str, str]:
        """使用正则从文本中提取四个段落的内容。

        Args:
            text: 包含四段式标记的完整解析文本。

        Returns:
            dict[str, str]: 段落标题到段落全文（含标题）的映射。
        """
        sections: Dict[str, str] = {}
        # 构建正则：匹配【标题】到下一个【标题】或字符串末尾
        pattern_parts = "|".join(re.escape(s) for s in self.required_sections)
        pattern = rf"({pattern_parts})(.*?)(?={pattern_parts}|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        for title, content in matches:
            sections[title] = title + content
        return sections

    def format_from_json(self, json_obj: dict) -> str:
        """将 LLM 返回的 JSON 对象组装为四段式文本。

        Args:
            json_obj: 包含 location, solution, options_analysis, warning 字段的字典。

        Returns:
            格式化的四段式文本。
        """
        parts: List[str] = []
        for field, section_title in JSON_FIELD_TO_SECTION.items():
            content = json_obj.get(field, "").strip()
            parts.append(f"{section_title}\n{content}")
        return "\n\n".join(parts)
