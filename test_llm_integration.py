"""LLM 集成模块统一测试文件。

覆盖: validator, checkpoint, config, result_merger, prompts,
       api_client (语法/导入), run_llm (CLI), qa_check, 跨模块导入。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# 切换到工作目录以确保相对导入正常
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


# ============================================================================
# Fixtures: 共用测试数据
# ============================================================================

VALID_FOUR_SECTION_TEXT = """【考点定位】第一章 世界的物质性及发展规律 - 对立统一规律

【解题思路】
第一步：审题。题干关键词是"矛盾"，这是唯物辩证法的核心概念。
第二步：回顾原理。对立统一规律揭示了事物发展的根本动力在于事物内部的矛盾性。矛盾的同一性和斗争性相互联结、相互制约，共同推动事物发展。
第三步：综合以上分析，确定正确答案为A。

【选项分析】
A项：正确。该选项准确表述了对立统一规律的核心内容，符合教材原文。
B项：错误。该选项将矛盾的斗争性绝对化，忽视了同一性的作用。与A项对比，犯了以偏概全的错误。
C项：错误。该选项属于形而上学观点，否认了矛盾的普遍性。
D项：错误。该选项混淆了主要矛盾和次要矛盾的关系。

【易错提醒】
本题最易混淆的是矛盾的同一性和斗争性的关系。注意口诀："同一斗争不可分，片面夸大必失分。"切忌将二者割裂开来或片面强调某一个方面。"""

SHORT_TEXT = "【考点定位】\n\n【解题思路】\n\n【选项分析】\n\n【易错提醒】"

MISSING_SECTION_TEXT = """【考点定位】第一章 世界的物质性及发展规律 - 对立统一规律

【解题思路】
第一步：审题。题干关键词是"矛盾"，这是唯物辩证法的核心概念。
第二步：回顾原理。对立统一规律揭示了事物发展的根本动力在于事物内部的矛盾性。

【选项分析】
A项：正确。该选项准确表述了对立统一规律的核心内容。
B项：错误。该选项将矛盾的斗争性绝对化。
C项：错误。该选项属于形而上学观点。
D项：错误。该选项混淆了主要矛盾和次要矛盾的关系。"""

# 缺失【易错提醒】
TEXT_MISSING_WARNING = """【考点定位】第一章 世界的物质性及发展规律 - 对立统一规律

【解题思路】
第一步：审题。题干关键词是"矛盾"。
第二步：回顾原理。对立统一规律揭示了事物发展的根本动力。
第三步：确定正确答案。

【选项分析】
A项：正确。该选项准确表述了对立统一规律的核心内容，符合教材原文。
B项：错误。与A项对比，犯了以偏概全的错误。
C项：错误。该选项属于形而上学观点。
D项：错误。混淆了主要矛盾和次要矛盾的关系。"""

# 各段内容不足20字符
SHORT_SECTION_TEXT = """【考点定位】第一章

【解题思路】
解题。

【选项分析】
A正确。

【易错提醒】
注意。"""

# JSON 格式的解析数据
VALID_JSON_DATA = {
    "location": "第一章 世界的物质性及发展规律 - 对立统一规律",
    "solution": "第一步：审题。关键词是'矛盾'。第二步：回顾原理。对立统一规律揭示事物发展动力。第三步：确定答案为A。",
    "options_analysis": "A项正确。B项错误，犯了以偏概全的错误。C项错误，属于形而上学观点。D项错误，混淆了概念。",
    "warning": "注意区分矛盾的同一性和斗争性，不能片面强调一个方面。典型错误是认为斗争性是唯一动力。",
}

SAMPLE_QUESTION = {
    "id": 1,
    "type": "single",
    "chapter": "第一章 世界的物质性及发展规律",
    "stem": "对立统一规律揭示了事物发展的（  ）",
    "options": {
        "A": "根本动力",
        "B": "唯一原因",
        "C": "外在力量",
        "D": "偶然现象",
    },
    "answer": "A",
    "difficulty": "easy",
}

SAMPLE_QUESTIONS_LIST = [
    SAMPLE_QUESTION,
    {
        "id": 2,
        "type": "multiple",
        "chapter": "第一章 世界的物质性及发展规律",
        "stem": "哲学基本问题包括（  ）",
        "options": {"A": "思维和存在何者为第一性", "B": "世界如何发展", "C": "思维能否认识现实世界", "D": "世界本质的数量"},
        "answer": "A,C",
    },
    {
        "id": 3,
        "type": "judge",
        "chapter": "绪论 马克思主义是中国共产党必须长期坚持的指导思想",
        "stem": "马克思主义是科学的世界观和方法论。",
        "options": {"A": "正确", "B": "错误"},
        "answer": "A",
    },
]

FEWSHOT_EXPLANATION_SINGLE = """【考点定位】第一章 世界的物质性及发展规律 - 哲学基本问题

【解题思路】
第一步：审题。题干关键词是"思维与存在的关系问题"。
第二步：回顾原理。恩格斯明确指出全部哲学的基本问题是思维和存在的关系问题。
第三步：得出答案C。

【选项分析】
A项：错误，范围缩小。B项：错误，范围缩小。C项：正确。D项：错误，范围缩小。

【易错提醒】
勿将哲学基本问题等同于某一学派的问题，所有哲学流派都必须回答。"""


# ============================================================================
# 1. TestOutputValidator — validator.py
# ============================================================================

class TestOutputValidator(unittest.TestCase):
    """测试 OutputValidator 核心校验逻辑。"""

    def setUp(self):
        self.validator = __import__("validator").OutputValidator()

    # --- validate() 测试 ---

    def test_validate_valid_four_section_text(self):
        """四段式完整文本 → is_valid=True"""
        from validator import OutputValidator
        v = OutputValidator()
        result = v.validate(VALID_FOUR_SECTION_TEXT)
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error_message}")
        self.assertEqual(result.missing_sections, [])
        self.assertGreater(result.total_length, 200)

    def test_validate_missing_section(self):
        """缺失段落 → is_valid=False, missing_sections 包含缺失段"""
        from validator import OutputValidator
        v = OutputValidator()
        result = v.validate(TEXT_MISSING_WARNING)
        self.assertFalse(result.is_valid)
        self.assertIn("【易错提醒】", result.missing_sections)

    def test_validate_total_length_too_short(self):
        """总长度 < 200 → is_valid=False（每段足够20字但总量不足200）"""
        from validator import OutputValidator
        v = OutputValidator(min_section_length=5, min_total_length=200)
        short = (
            "【考点定位】第一章 世界的物质性及发展规律\n"
            "【解题思路】第一步审题第二步回顾原理第三步得出结论\n"
            "【选项分析】A正确B错误C正确D错误\n"
            "【易错提醒】注意区分容易混淆的核心概念"
        )
        result = v.validate(short)
        self.assertFalse(result.is_valid)
        self.assertIn("总长度", result.error_message)

    def test_validate_section_content_too_short(self):
        """每段 < 20 字符有效内容 → is_valid=False"""
        from validator import OutputValidator
        v = OutputValidator()
        result = v.validate(SHORT_SECTION_TEXT)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.missing_sections), 0)

    def test_validate_empty_string(self):
        """空字符串 → is_valid=False, 所有段落缺失"""
        from validator import OutputValidator
        v = OutputValidator()
        result = v.validate("")
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.missing_sections), 4)
        self.assertIn("解析文本为空", result.error_message)

    def test_validate_whitespace_only(self):
        """纯空格 → is_valid=False"""
        from validator import OutputValidator
        v = OutputValidator()
        result = v.validate("   \n  \t  ")
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.missing_sections), 4)

    def test_validate_only_one_section(self):
        """只有1段 → is_valid=False，其余3段缺失"""
        from validator import OutputValidator
        v = OutputValidator()
        text = "【考点定位】第一章 世界的物质性及发展规律 - 对立统一规律\n\n这是关于对立统一规律的详细考点说明，包含足够多的内容来通过字数检查但是缺少其他必需段落。"
        result = v.validate(text)
        self.assertFalse(result.is_valid)
        self.assertIn("【解题思路】", result.missing_sections)
        self.assertIn("【选项分析】", result.missing_sections)
        self.assertIn("【易错提醒】", result.missing_sections)

    def test_validate_custom_min_lengths(self):
        """自定义 min_section_length 和 min_total_length 正确生效"""
        from validator import OutputValidator
        v = OutputValidator(min_section_length=5, min_total_length=10)
        minimal = "【考点定位】abcde\n【解题思路】fghij\n【选项分析】klmno\n【易错提醒】pqrst"
        result = v.validate(minimal)
        self.assertTrue(result.is_valid)

    # --- extract_sections() 测试 ---

    def test_extract_sections_all_present(self):
        """正则提取四个段落全部存在"""
        from validator import OutputValidator
        v = OutputValidator()
        sections = v.extract_sections(VALID_FOUR_SECTION_TEXT)
        self.assertEqual(len(sections), 4)
        for sec in v.required_sections:
            self.assertIn(sec, sections, f"Missing section: {sec}")

    def test_extract_sections_partial(self):
        """正则提取缺失段落"""
        from validator import OutputValidator
        v = OutputValidator()
        sections = v.extract_sections(TEXT_MISSING_WARNING)
        self.assertNotIn("【易错提醒】", sections)
        self.assertIn("【考点定位】", sections)
        self.assertIn("【解题思路】", sections)
        self.assertIn("【选项分析】", sections)

    def test_extract_sections_empty_string(self):
        """提取空字符串 → 空dict"""
        from validator import OutputValidator
        v = OutputValidator()
        sections = v.extract_sections("")
        self.assertEqual(sections, {})

    def test_extract_sections_no_markers(self):
        """无段落标记的文本 → 空dict"""
        from validator import OutputValidator
        v = OutputValidator()
        sections = v.extract_sections("这是一段没有标记的普通文本。")
        self.assertEqual(sections, {})

    # --- format_from_json() 测试 ---

    def test_format_from_json_complete(self):
        """JSON→四段式组装，所有字段完整"""
        from validator import OutputValidator
        v = OutputValidator()
        formatted = v.format_from_json(VALID_JSON_DATA)
        self.assertIn("【考点定位】", formatted)
        self.assertIn("【解题思路】", formatted)
        self.assertIn("【选项分析】", formatted)
        self.assertIn("【易错提醒】", formatted)
        # 验证通过二段换行分隔
        parts = formatted.split("\n\n")
        self.assertEqual(len(parts), 4)

    def test_format_from_json_missing_fields(self):
        """JSON 缺少字段 → 空内容段落"""
        from validator import OutputValidator
        v = OutputValidator()
        incomplete = {"location": "第一章", "solution": "解题"}
        formatted = v.format_from_json(incomplete)
        self.assertIn("【考点定位】", formatted)
        self.assertIn("【选项分析】\n", formatted)  # 段落标题存在但内容为空
        self.assertIn("【易错提醒】\n", formatted)

    def test_format_from_json_empty_dict(self):
        """空 JSON → 只有标题的文本"""
        from validator import OutputValidator
        v = OutputValidator()
        formatted = v.format_from_json({})
        # 每个段落标题都存在
        for title in v.required_sections:
            self.assertIn(title, formatted)

    # --- ValidationResult dataclass 测试 ---

    def test_validation_result_defaults(self):
        """ValidationResult dataclass 默认值"""
        from validator import ValidationResult
        vr = ValidationResult(is_valid=False)
        self.assertFalse(vr.is_valid)
        self.assertEqual(vr.missing_sections, [])
        self.assertEqual(vr.total_length, 0)
        self.assertEqual(vr.error_message, "")


# ============================================================================
# 2. TestCheckpointManager — checkpoint.py
# ============================================================================

class TestCheckpointManager(unittest.TestCase):
    """测试 CheckpointManager 断点续传逻辑。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.tmpdir, "checkpoint.json")
        from checkpoint import CheckpointManager
        self.mgr = CheckpointManager(checkpoint_path=self.checkpoint_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        """save() 写入文件，load() 读取成功"""
        from checkpoint import Checkpoint
        cp = Checkpoint(
            completed_ids={1, 2, 3},
            failed_ids={4},
            current_index=4,
            total_questions=100,
            model_used="glm-4-flash",
        )
        self.mgr.save(cp)
        self.assertTrue(os.path.exists(self.checkpoint_path))

        loaded = self.mgr.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.completed_ids, {1, 2, 3})
        self.assertEqual(loaded.failed_ids, {4})
        self.assertEqual(loaded.current_index, 4)
        self.assertEqual(loaded.total_questions, 100)
        self.assertEqual(loaded.model_used, "glm-4-flash")

    def test_load_nonexistent_file(self):
        """文件不存在 → load() 返回 None"""
        loaded = self.mgr.load()
        self.assertIsNone(loaded)

    def test_load_corrupted_file(self):
        """损坏的 JSON 文件 → load() 返回 None"""
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")
        loaded = self.mgr.load()
        self.assertIsNone(loaded)

    def test_mark_completed(self):
        """mark_completed() 正确更新状态"""
        from checkpoint import Checkpoint
        cp = Checkpoint(
            completed_ids={1},
            failed_ids={2},
            current_index=2,
            total_questions=100,
        )
        self.mgr.checkpoint = cp
        self.mgr.mark_completed(3)
        self.assertIn(3, cp.completed_ids)
        self.assertEqual(cp.current_index, 3)

    def test_mark_completed_removes_from_failed(self):
        """mark_completed() 从 failed_ids 中移除"""
        from checkpoint import Checkpoint
        cp = Checkpoint(
            completed_ids=set(),
            failed_ids={5},
            current_index=1,
            total_questions=100,
        )
        self.mgr.checkpoint = cp
        self.mgr.mark_completed(5)
        self.assertIn(5, cp.completed_ids)
        self.assertNotIn(5, cp.failed_ids)
        self.assertEqual(cp.current_index, 2)

    def test_mark_failed(self):
        """mark_failed() 正确更新状态"""
        from checkpoint import Checkpoint
        cp = Checkpoint(
            completed_ids={1},
            failed_ids=set(),
            current_index=1,
            total_questions=100,
        )
        self.mgr.checkpoint = cp
        self.mgr.mark_failed(2)
        self.assertIn(2, cp.failed_ids)
        self.assertEqual(cp.current_index, 2)

    def test_mark_on_none_checkpoint(self):
        """checkpoint 为 None 时 mark 操作不报错"""
        self.mgr.checkpoint = None
        # 不应抛出异常
        self.mgr.mark_completed(1)
        self.mgr.mark_failed(2)

    def test_get_pending_indices_no_checkpoint(self):
        """无 checkpoint → 返回全部题目"""
        pending = self.mgr.get_pending_indices(SAMPLE_QUESTIONS_LIST)
        self.assertEqual(len(pending), 3)

    def test_get_pending_indices_with_checkpoint(self):
        """有 checkpoint → 过滤已完成和失败题目"""
        from checkpoint import Checkpoint
        cp = Checkpoint(
            completed_ids={1},
            failed_ids={2},
            current_index=2,
            total_questions=3,
        )
        self.mgr.checkpoint = cp
        pending = self.mgr.get_pending_indices(SAMPLE_QUESTIONS_LIST)
        # 仅 id=3 未处理
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0][1]["id"], 3)

    def test_get_pending_indices_all_processed(self):
        """全部已处理 → 返回空列表"""
        from checkpoint import Checkpoint
        cp = Checkpoint(
            completed_ids={1, 2, 3},
            failed_ids=set(),
            current_index=3,
            total_questions=3,
        )
        self.mgr.checkpoint = cp
        pending = self.mgr.get_pending_indices(SAMPLE_QUESTIONS_LIST)
        self.assertEqual(pending, [])

    def test_atomic_write_no_tmp_leftover(self):
        """原子写入：.tmp 文件不存在于最终位置"""
        from checkpoint import Checkpoint
        cp = Checkpoint(completed_ids={1}, total_questions=10)
        self.mgr.save(cp)
        # .tmp 文件应该已被 os.replace 替换
        tmp_path = self.checkpoint_path + ".tmp"
        self.assertFalse(
            os.path.exists(tmp_path),
            f"临时文件 {tmp_path} 不应存在（原子写入应已替换）"
        )
        self.assertTrue(os.path.exists(self.checkpoint_path))

    def test_save_with_custom_path(self):
        """save() 自定义路径"""
        from checkpoint import Checkpoint
        custom_path = os.path.join(self.tmpdir, "custom_checkpoint.json")
        cp = Checkpoint(completed_ids={1}, total_questions=10)
        self.mgr.save(cp, path=custom_path)
        self.assertTrue(os.path.exists(custom_path))

    def test_init_checkpoint(self):
        """init_checkpoint() 创建正确的 checkpoint"""
        cp = self.mgr.init_checkpoint(
            total_questions=100,
            model_used="glm-4-flash",
            completed_ids={1, 2},
        )
        self.assertEqual(cp.total_questions, 100)
        self.assertEqual(cp.model_used, "glm-4-flash")
        self.assertEqual(cp.completed_ids, {1, 2})
        self.assertEqual(cp.current_index, 2)  # 2 个 completed
        self.assertIsNotNone(cp.started_at)

    def test_checkpoint_to_dict_from_dict_roundtrip(self):
        """Checkpoint.to_dict() ↔ from_dict() 往返一致"""
        from checkpoint import Checkpoint
        original = Checkpoint(
            completed_ids={1, 5, 10},
            failed_ids={3},
            current_index=4,
            total_questions=20,
            model_used="deepseek-chat",
            started_at="2024-01-01T00:00:00+00:00",
            last_updated="2024-01-01T01:00:00+00:00",
        )
        data = original.to_dict()
        restored = Checkpoint.from_dict(data)
        self.assertEqual(restored.completed_ids, original.completed_ids)
        self.assertEqual(restored.failed_ids, original.failed_ids)
        self.assertEqual(restored.current_index, original.current_index)
        self.assertEqual(restored.total_questions, original.total_questions)
        self.assertEqual(restored.model_used, original.model_used)

    def test_to_dict_sorts_ids(self):
        """to_dict() 对 IDs 排序"""
        from checkpoint import Checkpoint
        cp = Checkpoint(
            completed_ids={5, 1, 10},
            failed_ids={3, 7},
        )
        data = cp.to_dict()
        self.assertEqual(data["completed_ids"], [1, 5, 10])
        self.assertEqual(data["failed_ids"], [3, 7])


# ============================================================================
# 3. TestConfig — config.py
# ============================================================================

class TestModelConfig(unittest.TestCase):
    """测试 ModelConfig 默认值。"""

    def test_modelconfig_defaults(self):
        """ModelConfig 默认字段值正确"""
        from config import ModelConfig
        mc = ModelConfig(
            model_name="glm-4-flash",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key="test-key",
        )
        self.assertEqual(mc.model_name, "glm-4-flash")
        self.assertEqual(mc.temperature, 0.3)
        self.assertEqual(mc.max_tokens, 2000)
        self.assertEqual(mc.request_interval, 2.0)

    def test_modelconfig_custom_values(self):
        """ModelConfig 自定义值"""
        from config import ModelConfig
        mc = ModelConfig(
            model_name="custom-model",
            base_url="https://custom.api/v1",
            api_key="custom-key",
            temperature=0.7,
            max_tokens=4000,
            request_interval=5.0,
        )
        self.assertEqual(mc.temperature, 0.7)
        self.assertEqual(mc.max_tokens, 4000)
        self.assertEqual(mc.request_interval, 5.0)

    def test_preset_zhipu(self):
        """PRESET_ZHIPU 预置值"""
        from config import PRESET_ZHIPU
        self.assertEqual(PRESET_ZHIPU["model_name"], "glm-4-flash")
        self.assertIn("open.bigmodel.cn", PRESET_ZHIPU["base_url"])

    def test_preset_deepseek(self):
        """PRESET_DEEPSEEK 预置值"""
        from config import PRESET_DEEPSEEK
        self.assertEqual(PRESET_DEEPSEEK["model_name"], "deepseek-chat")
        self.assertIn("deepseek.com", PRESET_DEEPSEEK["base_url"])


class TestAppConfig(unittest.TestCase):
    """测试 AppConfig.load() 配置加载。"""

    def setUp(self):
        # 保存原始环境变量
        self.original_env = {}
        for key in [
            "ACTIVE_PROVIDER", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY",
            "TEMPERATURE", "MAX_TOKENS", "LLM_INPUT_FILE",
            "LLM_OUTPUT_FILE", "LLM_CHECKPOINT_FILE",
            "LLM_MAX_RETRIES", "LLM_MIN_EXPLANATION_LENGTH",
            "LLM_REQUEST_TIMEOUT",
        ]:
            self.original_env[key] = os.environ.get(key)

    def tearDown(self):
        # 恢复环境变量
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _clear_env(self):
        for key in [
            "ACTIVE_PROVIDER", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY",
            "TEMPERATURE", "MAX_TOKENS", "LLM_INPUT_FILE",
            "LLM_OUTPUT_FILE", "LLM_CHECKPOINT_FILE",
            "LLM_MAX_RETRIES", "LLM_MIN_EXPLANATION_LENGTH",
            "LLM_REQUEST_TIMEOUT",
        ]:
            os.environ.pop(key, None)

    def test_load_default_provider_zhipu(self):
        """无 ACTIVE_PROVIDER 时默认使用智谱"""
        from config import AppConfig
        self._clear_env()
        os.environ["ZHIPU_API_KEY"] = "test-zhipu-key"
        config = AppConfig.load()
        self.assertEqual(config.active_provider, "zhipu")
        self.assertEqual(config.model.model_name, "glm-4-flash")

    def test_load_deepseek_provider(self):
        """ACTIVE_PROVIDER=deepseek 时使用 DeepSeek"""
        from config import AppConfig
        self._clear_env()
        os.environ["ACTIVE_PROVIDER"] = "deepseek"
        os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
        config = AppConfig.load()
        self.assertEqual(config.active_provider, "deepseek")
        self.assertEqual(config.model.model_name, "deepseek-chat")

    def test_load_missing_api_key_raises(self):
        """缺少 API Key → ValueError"""
        from config import AppConfig
        self._clear_env()
        with self.assertRaises(ValueError) as ctx:
            AppConfig.load()
        self.assertIn("ZHIPU_API_KEY", str(ctx.exception))

    def test_load_unknown_provider_raises(self):
        """未知 provider → ValueError"""
        from config import AppConfig
        self._clear_env()
        os.environ["ACTIVE_PROVIDER"] = "unknown"
        with self.assertRaises(ValueError) as ctx:
            AppConfig.load()
        self.assertIn("未知", str(ctx.exception))

    def test_load_custom_env_values(self):
        """自定义环境变量正确传递"""
        from config import AppConfig
        self._clear_env()
        os.environ["ZHIPU_API_KEY"] = "test-key"
        os.environ["TEMPERATURE"] = "0.5"
        os.environ["MAX_TOKENS"] = "3000"
        os.environ["LLM_MAX_RETRIES"] = "5"
        os.environ["LLM_REQUEST_TIMEOUT"] = "90"
        config = AppConfig.load()
        self.assertEqual(config.model.temperature, 0.5)
        self.assertEqual(config.model.max_tokens, 3000)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.request_timeout, 90)

    def test_load_custom_file_paths(self):
        """自定义文件路径正确传递"""
        from config import AppConfig
        self._clear_env()
        os.environ["ZHIPU_API_KEY"] = "test-key"
        os.environ["LLM_INPUT_FILE"] = "custom_input.json"
        os.environ["LLM_OUTPUT_FILE"] = "custom_output.json"
        os.environ["LLM_CHECKPOINT_FILE"] = "custom_cp.json"
        config = AppConfig.load()
        self.assertEqual(config.input_file, "custom_input.json")
        self.assertEqual(config.output_file, "custom_output.json")
        self.assertEqual(config.checkpoint_file, "custom_cp.json")

    def test_load_default_file_paths(self):
        """默认文件路径"""
        from config import AppConfig
        self._clear_env()
        os.environ["ZHIPU_API_KEY"] = "test-key"
        config = AppConfig.load()
        self.assertEqual(config.input_file, "questions.json")
        self.assertEqual(config.output_file, "questions_with_llm.json")
        self.assertEqual(config.checkpoint_file, "checkpoint.json")

    def test_load_default_max_retries(self):
        """默认 max_retries=3"""
        from config import AppConfig
        self._clear_env()
        os.environ["ZHIPU_API_KEY"] = "test-key"
        config = AppConfig.load()
        self.assertEqual(config.max_retries, 3)

    def test_load_default_min_explanation_length(self):
        """默认 min_explanation_length=200"""
        from config import AppConfig
        self._clear_env()
        os.environ["ZHIPU_API_KEY"] = "test-key"
        config = AppConfig.load()
        self.assertEqual(config.min_explanation_length, 200)

    def test_load_default_request_timeout(self):
        """默认 request_timeout=60"""
        from config import AppConfig
        self._clear_env()
        os.environ["ZHIPU_API_KEY"] = "test-key"
        config = AppConfig.load()
        self.assertEqual(config.request_timeout, 60)

    def test_require_env_value_error_format(self):
        """_require_env 的错误信息格式"""
        from config import _require_env
        self._clear_env()
        with self.assertRaises(ValueError) as ctx:
            _require_env("NONEXISTENT_VAR", "测试描述")
        self.assertIn("NONEXISTENT_VAR", str(ctx.exception))
        self.assertIn("测试描述", str(ctx.exception))


# ============================================================================
# 4. TestResultMerger — result_merger.py
# ============================================================================

class TestResultMerger(unittest.TestCase):
    """测试结果合并模块。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_results_adds_explanations(self):
        """merge_results 正确添加 llm_explanation 字段"""
        from result_merger import merge_results
        output_path = os.path.join(self.tmpdir, "output.json")
        explanations = {1: "解析内容1", 2: "解析内容2"}
        merged_path = merge_results(SAMPLE_QUESTIONS_LIST, explanations, output_path)
        self.assertTrue(os.path.exists(merged_path))
        with open(merged_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["llm_explanation"], "解析内容1")
        self.assertEqual(data[1]["llm_explanation"], "解析内容2")

    def test_merge_results_empty_explanations_gets_empty_string(self):
        """无解析的题目 → llm_explanation 为空字符串"""
        from result_merger import merge_results
        output_path = os.path.join(self.tmpdir, "output.json")
        explanations = {1: "解析1"}
        merged_path = merge_results(SAMPLE_QUESTIONS_LIST, explanations, output_path)
        with open(merged_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[2]["llm_explanation"], "")  # id=3 no explanation

    def test_merge_results_does_not_modify_original(self):
        """merge_results 不修改原始题目列表"""
        from result_merger import merge_results
        original_copy = [dict(q) for q in SAMPLE_QUESTIONS_LIST]
        output_path = os.path.join(self.tmpdir, "output.json")
        explanations = {1: "解析1"}
        merge_results(SAMPLE_QUESTIONS_LIST, explanations, output_path)
        # 原始列表不应有 llm_explanation
        self.assertNotIn("llm_explanation", SAMPLE_QUESTIONS_LIST[0])

    def test_backup_original_creates_bak(self):
        """backup_original 创建 .bak 文件"""
        from result_merger import backup_original
        src_path = os.path.join(self.tmpdir, "original.json")
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_QUESTIONS_LIST, f)
        bak_path = backup_original(src_path)
        self.assertTrue(os.path.exists(bak_path))
        self.assertTrue(bak_path.endswith(".bak"))

    def test_backup_original_file_not_found(self):
        """backup_original 对不存在的文件抛出 FileNotFoundError"""
        from result_merger import backup_original
        with self.assertRaises(FileNotFoundError):
            backup_original(os.path.join(self.tmpdir, "nonexistent.json"))

    def test_merge_results_in_place_with_backup(self):
        """in_place 模式：先备份再覆盖原文件"""
        from result_merger import merge_results
        src_path = os.path.join(self.tmpdir, "original.json")
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_QUESTIONS_LIST, f, ensure_ascii=False)

        explanations = {1: "解析1"}
        result = merge_results(
            SAMPLE_QUESTIONS_LIST, explanations,
            output_path=os.path.join(self.tmpdir, "out.json"),
            in_place=True, input_path=src_path,
        )
        # 应返回原文件路径
        self.assertEqual(result, src_path)
        # .bak 文件应存在
        self.assertTrue(os.path.exists(src_path + ".bak"))

    def test_build_summary_basic(self):
        """build_summary 正确统计"""
        from result_merger import build_summary
        summary = build_summary(
            questions=SAMPLE_QUESTIONS_LIST,
            explanations={1: "解析1", 3: "解析3"},
            total_time=120.5,
            estimated_tokens=5000,
            failed_ids=[2],
        )
        self.assertEqual(summary["total_questions"], 3)
        self.assertEqual(summary["succeeded"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertIn("%", summary["success_rate"])
        self.assertEqual(summary["total_time_seconds"], 120.5)
        self.assertEqual(summary["estimated_token_consumption"], 5000)
        self.assertEqual(summary["failed_ids"], [2])

    def test_build_summary_empty_questions(self):
        """空题目列表：build_summary 不除零"""
        from result_merger import build_summary
        summary = build_summary(
            questions=[],
            explanations={},
            total_time=0,
            estimated_tokens=0,
        )
        self.assertEqual(summary["total_questions"], 0)
        self.assertEqual(summary["success_rate"], "0%")

    def test_build_summary_zero_succeeded(self):
        """零成功：avg_time 为 0"""
        from result_merger import build_summary
        summary = build_summary(
            questions=SAMPLE_QUESTIONS_LIST,
            explanations={},
            total_time=100,
            estimated_tokens=0,
        )
        self.assertEqual(summary["avg_time_per_question"], 0)

    def test_build_summary_has_generated_at(self):
        """build_summary 包含 generated_at 字段"""
        from result_merger import build_summary
        summary = build_summary(
            questions=SAMPLE_QUESTIONS_LIST,
            explanations={1: "a"},
            total_time=10,
            estimated_tokens=100,
        )
        self.assertIn("generated_at", summary)


# ============================================================================
# 5. TestPromptManager — prompts.py
# ============================================================================

class TestPromptManager(unittest.TestCase):
    """测试 PromptManager。"""

    @classmethod
    def setUpClass(cls):
        """确保 generate_explanations 模块可导入"""
        from generate_explanations import CHAPTER_KNOWLEDGE
        cls.chapter_knowledge = CHAPTER_KNOWLEDGE

    def test_fewshot_examples_pass_validator(self):
        """3个 few-shot 示例通过 validator 自检"""
        from prompts import _validate_all_examples
        errors = _validate_all_examples()
        self.assertEqual(
            errors, [],
            f"Few-shot examples failed validation: {errors}"
        )

    def test_prompt_manager_init_validates_examples(self):
        """PromptManager 初始化时自检 few-shot 示例"""
        from prompts import PromptManager
        pm = PromptManager()  # 不应抛出异常
        self.assertIsNotNone(pm)

    def test_build_messages_format(self):
        """build_messages 返回正确的消息列表格式"""
        from prompts import PromptManager
        pm = PromptManager()
        messages = pm.build_messages(SAMPLE_QUESTION)

        # 第一条是 system
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("content", messages[0])

        # 之后是 few-shot 示例 (user, assistant) 交替
        # 检查有 user 和 assistant 角色
        roles = [m["role"] for m in messages]
        self.assertIn("system", roles)
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

        # 最后一条是当前题目的 user 消息
        self.assertEqual(messages[-1]["role"], "user")

    def test_build_messages_includes_current_question(self):
        """build_messages 最后一条包含当前题目内容"""
        from prompts import PromptManager
        pm = PromptManager()
        messages = pm.build_messages(SAMPLE_QUESTION)
        last_content = messages[-1]["content"]
        self.assertIn("对立统一规律", last_content)
        self.assertIn("根本动力", last_content)

    def test_inject_knowledge_context_with_chapter(self):
        """inject_knowledge_context 根据 chapter 返回对应知识"""
        from prompts import PromptManager
        pm = PromptManager()
        knowledge = pm.inject_knowledge_context(SAMPLE_QUESTION)
        # 第一章 世界的物质性及发展规律 应有知识内容
        self.assertIsInstance(knowledge, str)
        self.assertGreater(len(knowledge), 0)

    def test_inject_knowledge_context_empty_chapter(self):
        """空章节 → 返回空字符串"""
        from prompts import PromptManager
        pm = PromptManager()
        question = {"chapter": ""}
        knowledge = pm.inject_knowledge_context(question)
        self.assertEqual(knowledge, "")

    def test_inject_knowledge_context_no_chapter(self):
        """无 chapter 字段 → 返回空字符串"""
        from prompts import PromptManager
        pm = PromptManager()
        question = {"stem": "test"}
        knowledge = pm.inject_knowledge_context(question)
        self.assertEqual(knowledge, "")

    def test_inject_knowledge_context_unknown_chapter(self):
        """未知章节 → 返回空字符串"""
        from prompts import PromptManager
        pm = PromptManager()
        question = {"chapter": "不存在的章节名称"}
        knowledge = pm.inject_knowledge_context(question)
        self.assertEqual(knowledge, "")

    def test_build_fallback_explanation_structure(self):
        """降级模板结构完整（含四个段落标题）"""
        from prompts import PromptManager
        pm = PromptManager()
        fallback = pm.build_fallback_explanation(SAMPLE_QUESTION)
        self.assertIn("【考点定位】", fallback)
        self.assertIn("【解题思路】", fallback)
        self.assertIn("【选项分析】", fallback)
        self.assertIn("【易错提醒】", fallback)

    def test_build_fallback_explanation_contains_answer(self):
        """降级模板包含正确答案"""
        from prompts import PromptManager
        pm = PromptManager()
        fallback = pm.build_fallback_explanation(SAMPLE_QUESTION)
        self.assertIn("A", fallback)

    def test_build_fallback_explanation_contains_chapter(self):
        """降级模板包含章节名"""
        from prompts import PromptManager
        pm = PromptManager()
        fallback = pm.build_fallback_explanation(SAMPLE_QUESTION)
        self.assertIn("第一章", fallback)

    def test_build_user_message_format(self):
        """build_user_message 格式正确"""
        from prompts import PromptManager
        pm = PromptManager()
        msg = pm.build_user_message(SAMPLE_QUESTION)
        self.assertIn("题目类型", msg)
        self.assertIn("单项选择题", msg)
        self.assertIn("对立统一规律", msg)
        self.assertIn("正确答案：A", msg)

    def test_build_user_message_judge_type(self):
        """判断题类型的 build_user_message"""
        from prompts import PromptManager
        pm = PromptManager()
        judge_q = {
            "type": "judge",
            "stem": "测试判断",
            "options": {"A": "正确", "B": "错误"},
            "answer": "A",
        }
        msg = pm.build_user_message(judge_q)
        self.assertIn("判断题", msg)


# ============================================================================
# 6. TestAPIClient — api_client.py (语法/导入验证，不调真实API)
# ============================================================================

class TestAPIClient(unittest.TestCase):
    """测试 api_client.py 的类结构和导入正确性（不调用真实 API）。"""

    def test_llm_response_dataclass_fields(self):
        """LLMResponse dataclass 字段正确"""
        from api_client import LLMResponse
        resp = LLMResponse(
            raw_text="test response",
            model="glm-4-flash",
            usage_tokens=100,
            latency_seconds=1.5,
        )
        self.assertEqual(resp.raw_text, "test response")
        self.assertEqual(resp.model, "glm-4-flash")
        self.assertEqual(resp.usage_tokens, 100)
        self.assertEqual(resp.latency_seconds, 1.5)

    def test_llm_response_defaults(self):
        """LLMResponse 默认值"""
        from api_client import LLMResponse
        resp = LLMResponse(raw_text="", model="test")
        self.assertEqual(resp.usage_tokens, 0)
        self.assertEqual(resp.latency_seconds, 0.0)

    def test_llm_generation_error_instantiation(self):
        """LLMGenerationError 异常类可正常实例化"""
        from api_client import LLMGenerationError
        error = LLMGenerationError(
            message="测试错误",
            attempts=3,
            last_error=ValueError("原始错误"),
        )
        self.assertEqual(str(error), "测试错误")
        self.assertEqual(error.attempts, 3)
        self.assertIsInstance(error.last_error, ValueError)

    def test_llm_generation_error_is_exception(self):
        """LLMGenerationError 是 Exception 子类"""
        from api_client import LLMGenerationError
        self.assertTrue(issubclass(LLMGenerationError, Exception))

    def test_api_client_instantiation(self):
        """APIClient 可正常实例化（openai 需要 base_url 和 api_key）"""
        from config import ModelConfig
        from api_client import APIClient
        mc = ModelConfig(
            model_name="glm-4-flash",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key="test-key",
        )
        client = APIClient(config=mc, max_retries=3)
        self.assertEqual(client.max_retries, 3)
        self.assertEqual(client.config.model_name, "glm-4-flash")
        self.assertEqual(client.total_calls, 0)
        self.assertEqual(client.total_tokens, 0)

    def test_api_client_has_total_retries(self):
        """APIClient 有重试计数器"""
        from config import ModelConfig
        from api_client import APIClient
        mc = ModelConfig(
            model_name="glm-4-flash",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key="test-key",
        )
        client = APIClient(config=mc)
        self.assertEqual(client.total_retries, 0)

    def test_llm_response_dataclass_is_dataclass(self):
        """LLMResponse 是 dataclass"""
        from dataclasses import is_dataclass
        from api_client import LLMResponse
        self.assertTrue(is_dataclass(LLMResponse))


# ============================================================================
# 7. TestRunLLM — run_llm.py (CLI 验证)
# ============================================================================

class TestRunLLM(unittest.TestCase):
    """测试 run_llm.py CLI 入口和工具函数。"""

    def test_parse_llm_json_pure_json(self):
        """parse_llm_json 解析纯 JSON 对象"""
        from run_llm import parse_llm_json
        text = '{"location":"考点","solution":"思路","options_analysis":"分析","warning":"提醒"}'
        result = parse_llm_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "考点")
        self.assertEqual(result["solution"], "思路")

    def test_parse_llm_json_markdown_code_block(self):
        """parse_llm_json 解析 Markdown 代码块中的 JSON"""
        from run_llm import parse_llm_json
        text = '```json\n{"location":"考点","solution":"思路","options_analysis":"分析","warning":"提醒"}\n```'
        result = parse_llm_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "考点")

    def test_parse_llm_json_partial_extraction(self):
        """parse_llm_json 正则提取四段内容（兜底）"""
        from run_llm import parse_llm_json
        text = """location: 第一章考点
solution: 解题思路内容
options_analysis: 选项分析内容
warning: 易错提醒内容"""
        result = parse_llm_json(text)
        self.assertIsNotNone(result)
        self.assertGreater(len(result["location"]), 0)
        self.assertGreater(len(result["solution"]), 0)

    def test_parse_llm_json_invalid(self):
        """parse_llm_json 无效文本 → None"""
        from run_llm import parse_llm_json
        text = "这不是 JSON，也不是任何可解析的格式。"
        result = parse_llm_json(text)
        self.assertIsNone(result)

    def test_parse_llm_json_empty_string(self):
        """parse_llm_json 空字符串 → None"""
        from run_llm import parse_llm_json
        result = parse_llm_json("")
        self.assertIsNone(result)

    def test_cli_help_runs(self):
        """python run_llm.py --help 正常执行"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "run_llm.py", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("马原题库", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--force", result.stdout)
        self.assertIn("--sample", result.stdout)

    def test_load_questions_sample(self):
        """load_questions 采样模式"""
        from run_llm import load_questions
        questions = load_questions("questions.json", sample=5)
        self.assertEqual(len(questions), 5)

    def test_load_questions_full(self):
        """load_questions 加载全部"""
        from run_llm import load_questions
        questions = load_questions("questions.json")
        self.assertGreater(len(questions), 100)


# ============================================================================
# 8. TestQACheck — qa_check.py
# ============================================================================

class TestQACheck(unittest.TestCase):
    """测试 qa_check.py 质量检查脚本。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sample_json_path = os.path.join(self.tmpdir, "test_questions.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_sample(self, questions):
        with open(self.sample_json_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False)

    def test_analyze_explanations_basic(self):
        """analyze_explanations 基本统计正确"""
        from qa_check import analyze_explanations
        questions = [
            {
                "id": 1, "type": "single", "chapter": "第一章",
                "stem": "test", "options": {}, "answer": "A",
                "llm_explanation": VALID_FOUR_SECTION_TEXT,
            },
            {
                "id": 2, "type": "single", "chapter": "第一章",
                "stem": "test", "options": {}, "answer": "B",
                "llm_explanation": "",
            },
        ]
        stats = analyze_explanations(questions)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["valid_count"], 1)
        self.assertEqual(stats["blank_or_abnormal_count"], 1)
        self.assertIn(2, stats["blank_or_abnormal_ids"])

    def test_analyze_explanations_all_valid(self):
        """全部有效解析"""
        from qa_check import analyze_explanations
        questions = [
            {
                "id": 1, "type": "single", "chapter": "第一章",
                "stem": "test", "options": {}, "answer": "A",
                "llm_explanation": VALID_FOUR_SECTION_TEXT,
            },
            {
                "id": 2, "type": "multiple", "chapter": "第一章",
                "stem": "test", "options": {}, "answer": "A,C",
                "llm_explanation": VALID_FOUR_SECTION_TEXT,
            },
        ]
        stats = analyze_explanations(questions)
        self.assertEqual(stats["valid_count"], 2)
        self.assertEqual(stats["blank_or_abnormal_count"], 0)

    def test_analyze_explanations_section_stats(self):
        """section_present 和 section_missing 统计正确"""
        from qa_check import analyze_explanations
        questions = [
            {
                "id": 1, "type": "single", "chapter": "第一章",
                "stem": "test", "options": {}, "answer": "A",
                "llm_explanation": VALID_FOUR_SECTION_TEXT,
            },
            {
                "id": 2, "type": "single", "chapter": "第一章",
                "stem": "test", "options": {}, "answer": "B",
                "llm_explanation": TEXT_MISSING_WARNING,
            },
        ]
        stats = analyze_explanations(questions)
        # 第一题有所有4段，第二题缺易错提醒
        self.assertGreaterEqual(stats["section_present"].get("【考点定位】", 0), 1)
        self.assertGreaterEqual(stats["section_missing"].get("【易错提醒】", 0), 1)

    def test_analyze_explanations_by_type(self):
        """按题型分组统计"""
        from qa_check import analyze_explanations
        questions = [
            {"id": 1, "type": "single", "chapter": "第一章", "stem": "t", "options": {}, "answer": "A",
             "llm_explanation": VALID_FOUR_SECTION_TEXT},
            {"id": 2, "type": "multiple", "chapter": "第一章", "stem": "t", "options": {}, "answer": "A,C",
             "llm_explanation": VALID_FOUR_SECTION_TEXT},
            {"id": 3, "type": "judge", "chapter": "第一章", "stem": "t", "options": {}, "answer": "A",
             "llm_explanation": VALID_FOUR_SECTION_TEXT},
        ]
        stats = analyze_explanations(questions)
        by_type = stats["by_type"]
        self.assertIn("single", by_type)
        self.assertIn("multiple", by_type)
        self.assertIn("judge", by_type)
        self.assertEqual(by_type["single"]["total"], 1)
        self.assertEqual(by_type["multiple"]["total"], 1)

    def test_analyze_explanations_by_chapter(self):
        """按章节分组统计"""
        from qa_check import analyze_explanations
        questions = [
            {"id": 1, "type": "single", "chapter": "第一章 A", "stem": "t", "options": {}, "answer": "A",
             "llm_explanation": VALID_FOUR_SECTION_TEXT},
            {"id": 2, "type": "single", "chapter": "第二章 B", "stem": "t", "options": {}, "answer": "B",
             "llm_explanation": VALID_FOUR_SECTION_TEXT},
        ]
        stats = analyze_explanations(questions)
        by_chapter = stats["by_chapter"]
        self.assertEqual(len(by_chapter), 2)

    def test_analyze_explanations_short_text_abnormal(self):
        """短于100字符的解析标记为异常"""
        from qa_check import analyze_explanations
        questions = [
            {"id": 1, "type": "single", "chapter": "第一章", "stem": "t", "options": {}, "answer": "A",
             "llm_explanation": "短文本"},
        ]
        stats = analyze_explanations(questions)
        self.assertIn(1, stats["blank_or_abnormal_ids"])

    def test_cli_help_runs(self):
        """python qa_check.py --help 正常执行"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "qa_check.py", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("质量检查", result.stdout)

    def test_display_sample(self):
        """display_sample 不崩溃"""
        from qa_check import display_sample
        questions = [
            {"id": i, "type": "single", "chapter": "第一章", "stem": f"题目{i}",
             "options": {"A": "选项A"}, "answer": "A",
             "llm_explanation": VALID_FOUR_SECTION_TEXT}
            for i in range(10)
        ]
        # 不应抛出异常
        display_sample(questions, n=3, random_seed=42)


# ============================================================================
# 9. TestCrossImports — 跨模块导入验证
# ============================================================================

class TestCrossImports(unittest.TestCase):
    """验证所有跨文件 import 正确。"""

    def test_prompts_imports_chapter_knowledge(self):
        """prompts.py → generate_explanations.CHAPTER_KNOWLEDGE"""
        from prompts import PromptManager
        pm = PromptManager()
        # CHAPTER_KNOWLEDGE 应可正常访问
        self.assertIsNotNone(pm)

    def test_run_llm_imports_all_modules(self):
        """run_llm.py → config, api_client, validator, checkpoint, prompts, result_merger"""
        # 直接 import run_llm 验证所有依赖
        import run_llm as rl
        self.assertTrue(hasattr(rl, "AppConfig"))
        self.assertTrue(hasattr(rl, "APIClient"))
        self.assertTrue(hasattr(rl, "OutputValidator"))
        self.assertTrue(hasattr(rl, "CheckpointManager"))
        self.assertTrue(hasattr(rl, "PromptManager"))
        self.assertTrue(hasattr(rl, "merge_results"))
        self.assertTrue(hasattr(rl, "build_summary"))

    def test_api_client_imports_config(self):
        """api_client.py → config.ModelConfig"""
        from api_client import APIClient
        from config import ModelConfig
        mc = ModelConfig(
            model_name="test",
            base_url="https://test.com/v1",
            api_key="key",
        )
        client = APIClient(config=mc)
        self.assertEqual(client.config, mc)

    def test_qa_check_imports_validator(self):
        """qa_check.py → validator.OutputValidator"""
        from qa_check import analyze_explanations
        self.assertTrue(callable(analyze_explanations))

    def test_circular_imports(self):
        """验证无循环导入"""
        import config
        import validator
        import checkpoint
        import prompts
        import api_client
        import result_merger
        import run_llm
        import qa_check
        self.assertTrue(True)  # 全部导入成功即可


# ============================================================================
# 10. TestDependencies — requirements.txt 依赖检查
# ============================================================================

class TestDependencies(unittest.TestCase):
    """验证 requirements.txt 中的依赖可安装/已安装。"""

    def test_openai_installed(self):
        """openai 包已安装"""
        import openai
        self.assertIsNotNone(openai.__version__)

    def test_tqdm_installed(self):
        """tqdm 包已安装"""
        import tqdm
        self.assertIsNotNone(tqdm.__version__)

    def test_requirements_txt_contents(self):
        """requirements.txt 包含 openai 和 tqdm"""
        with open("requirements.txt", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("openai", content)
        self.assertIn("tqdm", content)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
