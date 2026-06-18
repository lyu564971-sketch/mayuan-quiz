"""Prompt 管理模块。

管理系统 Prompt、Few-shot 示例和用户消息构建。
利用 CHAPTER_KNOWLEDGE 注入章节知识点。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from validator import OutputValidator

# 章节知识点字典（原 generate_explanations.CHAPTER_KNOWLEDGE 已内联）
# 后续可按需补充各章节的详细知识点内容
CHAPTER_KNOWLEDGE: Dict[str, Any] = {}


@dataclass
class FewShotExample:
    """Few-shot 示例。

    Attributes:
        question_type: 题型（single/multiple/judge）。
        question_stem: 题干文本。
        options: 选项字典。
        answer: 正确答案。
        explanation: 完整四段式解析文本。
    """

    question_type: str
    question_stem: str
    options: Dict[str, str]
    answer: str
    explanation: str


# ============================================================================
# 系统 Prompt
# ============================================================================

SYSTEM_PROMPT = """你是一位考研政治马克思主义基本原理（马原）教培专家。

## 任务
为考研政治马原题目生成四段式解析。

## 输出格式（严格遵守）
用【】标记分段，全文使用中文，禁止使用英文key或JSON：

【考点定位】
XX章 XX节 - XX知识点

【解题思路】
审题：指出题干关键词和考查知识点。
原理：阐述教材核心原理。
结论：运用原理解析题干，得出答案。

【选项分析】
A项：正确/错误。分析内容...
B项：正确/错误。分析内容...
（逐项分析，不跳过）

【易错提醒】
易错点和区分方法。

## 规则
1. 语言精练，直接分析，不写废话
2. 原理表述与教材一致
3. 每段30-100字
4. 禁止出现英文单词、JSON格式、step/option等英文标记
5. 选项分析中用"A项""B项"而非"A：correct""B：incorrect\""""


# ============================================================================
# Few-shot 示例（基于真实题目精心撰写）
# ============================================================================

# --- Q1: 单选题示例 ---
FEWSHOT_SINGLE = FewShotExample(
    question_type="single",
    question_stem="思维与存在的关系问题是（）",
    options={
        "A": "唯心主义哲学的基本问题",
        "B": "唯物主义哲学的基本问题",
        "C": "全部哲学的基本问题",
        "D": "一部分哲学的基本问题",
    },
    answer="C",
    explanation="""【考点定位】第一章 世界的物质性及发展规律 - 哲学基本问题

【解题思路】
第一步：审题。题干关键词是"思维与存在的关系问题"，这是哲学基本问题的标志性表述。哲学基本问题即思维和存在、精神和物质的关系问题。
第二步：回顾原理。恩格斯在《路德维希·费尔巴哈和德国古典哲学的终结》中明确指出："全部哲学，特别是近代哲学的重大的基本问题，是思维和存在的关系问题。"哲学基本问题包含两方面内容：第一方面是思维和存在何者为第一性（本体论问题），第二方面是思维能否正确认识存在（认识论问题）。对第一方面的不同回答划分了唯物主义和唯心主义；对第二方面的不同回答划分了可知论和不可知论。
第三步：得出结论。思维与存在的关系问题不是某一学派特有的问题，而是贯穿全部哲学发展史的根本问题，因此正确选项为C。

【选项分析】
A项：错误。该选项将"全部哲学的基本问题"缩小为"唯心主义哲学的基本问题"，犯了范围缩小的错误。思维与存在的关系问题同样也是唯物主义哲学必须回答的问题，并非唯心主义独有。与C项"全部哲学的基本问题"对比，该选项偷换概念。
B项：错误。该选项同样犯了范围缩小的错误。唯物主义哲学固然要回答思维与存在的关系问题，但唯心主义哲学也同样面对这一问题。哲学基本问题的普遍性决定了它不能局限于某一学派。与C项"全部哲学的基本问题"对比，该选项概念错误。
C项：正确。哲学基本问题是思维和存在的关系问题，这是贯穿全部哲学史的根本性问题，任何哲学流派都无法回避。恩格斯明确将其定位为"全部哲学"的基本问题。
D项：错误。该选项将"全部哲学"替换为"一部分哲学"，同样缩小了适用范围。哲学基本问题是所有哲学流派都必须回答的前提性问题，不是部分哲学才需要关注的。与C项"全部哲学的基本问题"对比，该选项范围缩小。

【易错提醒】
本题最易混淆的是将"哲学基本问题"等同于"唯物主义与唯心主义的区分标准"。注意：思维与存在的关系问题包含本体论和认识论两个层面，而"何者为第一性"仅是其本体论层面的一个方面。判别口诀："全部哲学最基本，两分维度不偏重。"此外，切忌因教材常以唯物主义视角展开，就误以为哲学基本问题只属于唯物主义。""",
)

# --- Q101: 多选题示例 ---
FEWSHOT_MULTIPLE = FewShotExample(
    question_type="multiple",
    question_stem="哲学基本问题主要包括两方面内容（   ）",
    options={
        "A": "思维和存在何者为第一性的问题",
        "B": "世界是如何发展的问题",
        "C": "我们的思维能不能认识现实世界",
        "D": "世界的本质是一个还是多个的问题",
    },
    answer="A,C",
    explanation="""【考点定位】第一章 世界的物质性及发展规律 - 哲学基本问题

【解题思路】
第一步：审题。题干关键词为"哲学基本问题主要包括两方面内容"，明确考查哲学基本问题的构成维度，且为多选题，需选出全部正确表述。
第二步：回顾原理。哲学基本问题是思维和存在的关系问题，包含两个方面：第一方面是思维和存在、精神和物质何者为第一性、何者为第二性的问题，即本体论问题——对这一问题的不同回答，划分了唯物主义和唯心主义两大阵营。第二方面是"我们关于我们周围世界的思想对这个世界本身的关系是怎样的？我们的思维能不能认识现实世界？我们能不能在我们关于现实世界的表象和概念中正确地反映现实？"即认识论问题——对这一问题的不同回答，划分了可知论和不可知论。
第三步：逐项判断。A项对应第一方面（本体论），正确。C项对应第二方面（认识论），正确。B项涉及"世界如何发展"，这属于辩证法与形而上学的分歧，不是哲学基本问题本身。D项涉及"世界的本质是一个还是多个"，这属于一元论与多元论的分歧，也不是哲学基本问题的两个维度。因此正确选项为A、C。

【选项分析】
A项：正确。这是哲学基本问题第一方面——本体论问题的标准表述。"思维和存在何者为第一性"即问物质和意识谁决定谁，这是划分唯物主义（物质第一性）和唯心主义（意识第一性）的根本标准。
B项：错误。世界如何发展（是否联系、是否变化、变化动因是什么）涉及的是辩证法和形而上学的分歧，而不是哲学基本问题的两个方面。哲学基本问题聚焦于思维与存在的关系，而非世界的发展状态。与A项和C项对比，该选项偷换概念。
C项：正确。这是哲学基本问题第二方面——认识论问题的标准表述。"我们的思维能不能认识现实世界"即问思维能否正确反映存在，这是划分可知论和不可知论的根本标准。
D项：错误。世界的本质是一个还是多个，涉及"一元论"与"多元论"的对立，属于世界本原数量问题，并非哲学基本问题的两个维度。与A项和C项对比，该选项概念混淆。

【易错提醒】
常见错误是将哲学基本问题与其他哲学分歧混为一谈。关键记忆：哲学基本问题 = 本体论（谁第一）+ 认识论（能否认识）；而"世界如何发展"对应辩证法/形而上学，"本质几个"对应一元论/多元论。多选题中需逐一对照这两个维度严格筛选，避免将其他哲学争论误归入哲学基本问题。口诀："基本问题两个面，本体认识一条线；发展本质另算事，不可混为一锅端。"此外，注意答案格式：多选题答案以选项字母组合表示，如"AC"等同于"A,C"格式，答题时均视为正确。""",
)

# --- Q481: 判断题示例 ---
FEWSHOT_JUDGE = FewShotExample(
    question_type="judge",
    question_stem="有一种观点认为，阶级性与科学性是不相容的，凡是代表某个阶级利益和愿望的社会理论，就不可能是科学的。因为马克思主义具有阶级性，所以是不科学的。",
    options={
        "A": "正确",
        "B": "错误",
    },
    answer="B",
    explanation="""【考点定位】绪论 马克思主义是关于无产阶级和人类解放的科学 - 马克思主义的鲜明特征

【解题思路】
第一步：审题。题干核心论点是"阶级性与科学性不相容"，并以此为前提推论"马克思主义因有阶级性故不科学"。这是一个三段论式的表述，需要判断其前提和推理是否正确。
第二步：回顾原理。马克思主义的鲜明特征之一是科学性与革命性（阶级性）的统一。马克思主义既具有严格的科学性——它揭示了自然界、人类社会和思维发展的普遍规律；又具有鲜明的阶级性——它公然申明是为无产阶级和广大劳动人民服务的理论。这两者并不矛盾，原因在于：无产阶级是最先进、最革命的阶级，其阶级利益与社会历史发展的客观规律高度一致。无产阶级只有解放全人类才能最终解放自己，因此无产阶级的利益追求与历史前进方向完全吻合。正因为如此，马克思主义的阶级性不但不排斥科学性，反而是其科学性得以保证的社会条件。
第三步：得出结论。题干的前提"阶级性与科学性不相容"本身是错误的，它看不到无产阶级阶级性与科学性内在统一的关系。因此，以其错误前提推导出的结论也是错误的。正确答案为B（错误）。

【选项分析】
A项（正确）：错误。如果选择"正确"则意味着认可"阶级性与科学性绝对对立"的错误观点。事实上，判断一种社会理论是否科学，标准在于它是否揭示了客观规律并被实践所检验，而非看它是否具有阶级性。马克思主义的科学性已由中国革命、建设和改革的伟大实践反复证明。
B项（错误）：正确。题干表述存在逻辑谬误——将无产阶级的阶级性简单等同于一般阶级性，忽视了无产阶级阶级性的特殊性。马克思主义的阶级性与科学性内在统一，因为无产阶级的利益与社会历史发展规律的方向一致。因此题干的观点是错误的，选择B。

【易错提醒】
本题最大陷阱在于题干以看似严密的三段论形式呈现，考生容易跟着题干的逻辑走而忽略前提本身的错误。关键区分："一般阶级性"可能遮蔽科学性，但"无产阶级阶级性"因其与历史规律一致性而能够与科学性统一。记忆要点：马克思主义 = 科学性 + 革命性（阶级性）+ 实践性，三者是有机统一的。典型错误思路是默认接受"凡是代表阶级利益的理论就不科学"这一前提，须知所有社会科学理论都有一定的价值立场，关键在于该立场是否符合客观历史发展趋势。""",
)

# 所有 few-shot 示例列表
FEWSHOT_EXAMPLES: List[FewShotExample] = [FEWSHOT_SINGLE]


# --- 降级模板 ---
FALLBACK_TEMPLATE = """【考点定位】
{chapter}相关知识点

【解题思路】
第一步：审题。题干关键词为"{stem_keywords}"，本题{question_type_desc}，考查{chapter}的核心原理。
第二步：回顾原理。{knowledge_fragment}
第三步：综合以上分析，确定正确答案为{answer}。

【选项分析】
{options_fallback}

【易错提醒】
本题涉及的{chapter}知识点中，需注意区分容易混淆的相近概念，建议结合教材原文进行对照学习。"""


def _validate_all_examples() -> List[str]:
    """自检所有 few-shot 示例是否通过四段式校验。

    Returns:
        错误信息列表，空列表表示全部通过。
    """
    validator = OutputValidator()
    errors: List[str] = []
    for i, example in enumerate(FEWSHOT_EXAMPLES):
        result = validator.validate(example.explanation)
        if not result.is_valid:
            errors.append(
                f"Few-shot 示例 #{i} ({example.question_type}) 校验失败: "
                f"{result.error_message}"
            )
    return errors


class PromptManager:
    """Prompt 管理器。

    负责构建发送给 LLM 的完整消息列表，包括系统 Prompt、
    Few-shot 示例和当前题目的用户消息。
    """

    def __init__(self):
        """初始化 Prompt 管理器，自动校验 few-shot 示例。"""
        self.validator = OutputValidator()
        self._check_examples()

    def _check_examples(self) -> None:
        """启动时自检 few-shot 示例质量。"""
        errors = _validate_all_examples()
        if errors:
            raise ValueError(
                "Few-shot 示例校验未通过:\n" + "\n".join(errors)
            )

    def build_user_message(self, question: Dict[str, Any]) -> str:
        """将题目 JSON 序列化为结构化的用户消息。

        Args:
            question: 题目字典，包含 stem, type, options, answer 等字段。

        Returns:
            格式化的用户消息字符串。
        """
        q_type = question.get("type", "single")
        stem = question.get("stem", "")
        options = question.get("options", {})
        answer = question.get("answer", "")
        chapter = question.get("chapter", "")

        type_map = {
            "single": "单项选择题",
            "multiple": "多项选择题",
            "judge": "判断题",
        }
        type_desc = type_map.get(q_type, "题目")

        parts = [
            f"题目类型：{type_desc}",
            f"所属章节：{chapter}" if chapter else "",
            f"题目：{stem}",
        ]

        if options:
            parts.append("选项：")
            for key in sorted(options.keys()):
                parts.append(f"  {key}. {options[key]}")

        parts.append(f"正确答案：{answer}")

        # 注入章节知识点
        knowledge = self.inject_knowledge_context(question)
        if knowledge:
            parts.append(f"\n参考知识点：\n{knowledge}")

        parts.append("\n请按照System Prompt要求生成四段式解析，必须使用【】分段，不要输出 JSON。")

        return "\n".join(p for p in parts if p)

    def build_messages(
        self, question: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """构建完整的消息列表，包含 system + few-shot + 当前问题。

        格式: [system, user_fewshot1, assistant_fewshot1,
               user_fewshot2, assistant_fewshot2, ...,
               user_current]

        Args:
            question: 当前题目字典。

        Returns:
            标准消息列表。
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # 插入 few-shot 示例
        for example in FEWSHOT_EXAMPLES:
            user_msg = self.build_user_message(
                {
                    "type": example.question_type,
                    "stem": example.question_stem,
                    "options": example.options,
                    "answer": example.answer,
                }
            )
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": example.explanation})

        # 当前题目
        current_msg = self.build_user_message(question)
        messages.append({"role": "user", "content": current_msg})

        return messages

    def inject_knowledge_context(self, question: Dict[str, Any]) -> str:
        """根据题目的 chapter 字段从 CHAPTER_KNOWLEDGE 提取对应知识点。

        Args:
            question: 题目字典。

        Returns:
            该章节的知识点文本，如果未找到则返回空字符串。
        """
        chapter = question.get("chapter", "")
        if not chapter:
            return ""

        knowledge = self._lookup_knowledge(chapter)
        if knowledge is None:
            return ""

        return self._flatten_knowledge(knowledge)

    @staticmethod
    def _lookup_knowledge(
        chapter: str,
    ) -> Optional[dict]:
        """在 CHAPTER_KNOWLEDGE 中查找章节。

        Args:
            chapter: 章节名称。

        Returns:
            知识点字典，未找到返回 None。
        """
        # 尝试精确匹配
        if chapter in CHAPTER_KNOWLEDGE:
            return CHAPTER_KNOWLEDGE[chapter]

        # 尝试前缀匹配
        for key, value in CHAPTER_KNOWLEDGE.items():
            if chapter.startswith(key) or key.startswith(chapter[:8]):
                return value

        # 尝试取章节前几个字匹配
        prefix = chapter[:6]
        for key, value in CHAPTER_KNOWLEDGE.items():
            if key.startswith(prefix):
                return value

        return None

    @staticmethod
    def _flatten_knowledge(knowledge: dict, indent: int = 0) -> str:
        """将嵌套知识点字典展平为可读的文本字符串。

        Args:
            knowledge: 嵌套知识点字典。
            indent: 当前缩进级别。

        Returns:
            格式化的文本字符串。
        """
        lines: List[str] = []
        prefix = "  " * indent
        for key, value in knowledge.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}### {key}")
                lines.append(PromptManager._flatten_knowledge(value, indent + 1))
            else:
                lines.append(f"{prefix}- {key}：{value}")
        return "\n".join(lines)

    def build_fallback_explanation(
        self, question: Dict[str, Any]
    ) -> str:
        """重试耗尽后使用知识库模板构建降级解析。

        Args:
            question: 题目字典。

        Returns:
            使用降级模板填充的四段式解析文本。
        """
        chapter = question.get("chapter", "未知章节")
        stem = question.get("stem", "")
        answer = question.get("answer", "")
        options = question.get("options", {})
        q_type = question.get("type", "single")

        type_map = {
            "single": "单项选择题",
            "multiple": "多项选择题",
            "judge": "判断题",
        }
        question_type_desc = type_map.get(q_type, "题目")

        # 题目关键词：取题干前20字
        stem_keywords = stem[:30] + ("..." if len(stem) > 30 else "")

        # 知识点片段
        knowledge = self.inject_knowledge_context(question)
        knowledge_fragment = (
            knowledge[:300] + "..."
            if len(knowledge) > 300
            else knowledge
        )
        if not knowledge_fragment:
            knowledge_fragment = f"请参考《马克思主义基本原理概论》中'{chapter}'的相关内容。"

        # 选项降级分析
        options_lines: List[str] = []
        for key in sorted(options.keys()):
            is_correct = False
            # 兼容多选答案 "AB" 和 "A,B" 格式
            answer_letters = set(answer.replace(",", "").strip())
            if key in answer_letters:
                is_correct = True

            if is_correct:
                options_lines.append(
                    f"{key}项：正确。该选项符合{chapter}中的相关原理。"
                )
            else:
                options_lines.append(
                    f"{key}项：错误。该选项与教材关于{chapter}的论述不符，"
                    f"建议对照教材原文辨析。"
                )
        options_fallback = "\n".join(options_lines)

        return FALLBACK_TEMPLATE.format(
            chapter=chapter,
            stem_keywords=stem_keywords,
            question_type_desc=question_type_desc,
            knowledge_fragment=knowledge_fragment,
            answer=answer,
            options_fallback=options_fallback,
        )


