"""db.py 单元测试 — FTS5 分词 + 向量序列化"""

import struct
import pytest

from robotmem.db import (
    tokenize_for_fts5,
    floats_to_blob,
    blob_to_floats,
    SUPPORTS_CONTENTLESS_DELETE,
    _CJK_RE,
)


class TestTokenizeForFts5:
    def test_pure_english(self):
        assert tokenize_for_fts5("hello world") == "hello world"

    def test_empty_string(self):
        assert tokenize_for_fts5("") == ""

    def test_none_input(self):
        """None → falsy → 原样返回"""
        assert tokenize_for_fts5(None) is None

    def test_cjk_detection(self):
        """CJK 字符正则检测"""
        assert _CJK_RE.search("中文") is not None
        assert _CJK_RE.search("hello") is None
        assert _CJK_RE.search("hello 中文 world") is not None

    def test_cjk_text_returns_string(self):
        """含 CJK 文本 — 不论 jieba 是否装都返回字符串"""
        result = tokenize_for_fts5("机器人抓取物体")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mixed_text(self):
        """中英混合 — 返回字符串"""
        result = tokenize_for_fts5("robot 抓取 task")
        assert isinstance(result, str)

    def test_japanese_cjk(self):
        """日文假名触发 CJK 分支"""
        assert _CJK_RE.search("こんにちは") is not None

    def test_korean_cjk(self):
        """韩文触发 CJK 分支"""
        assert _CJK_RE.search("안녕하세요") is not None


class TestFloatsToBlob:
    def test_basic(self):
        blob = floats_to_blob([1.0, 2.0, 3.0])
        assert isinstance(blob, bytes)
        assert len(blob) == 12  # 3 * 4 bytes

    def test_empty(self):
        blob = floats_to_blob([])
        assert blob == b""

    def test_single(self):
        blob = floats_to_blob([3.14])
        assert len(blob) == 4

    def test_roundtrip(self):
        """floats → blob → floats 往返"""
        original = [1.0, 2.5, -3.7, 0.0]
        blob = floats_to_blob(original)
        restored = blob_to_floats(blob, 4)
        for a, b in zip(original, restored):
            assert abs(a - b) < 1e-6


class TestBlobToFloats:
    def test_basic(self):
        blob = struct.pack("3f", 1.0, 2.0, 3.0)
        result = blob_to_floats(blob, 3)
        assert len(result) == 3
        assert abs(result[0] - 1.0) < 1e-6

    def test_dim_mismatch(self):
        """维度不匹配 → ValueError"""
        blob = struct.pack("2f", 1.0, 2.0)  # 8 bytes
        with pytest.raises(ValueError, match="向量维度不匹配"):
            blob_to_floats(blob, 3)  # 期望 12 bytes

    def test_empty_blob(self):
        result = blob_to_floats(b"", 0)
        assert result == []

    def test_single_dim(self):
        blob = struct.pack("1f", 42.0)
        result = blob_to_floats(blob, 1)
        assert abs(result[0] - 42.0) < 1e-6


class TestSupportsContentlessDelete:
    def test_is_bool(self):
        assert isinstance(SUPPORTS_CONTENTLESS_DELETE, bool)
