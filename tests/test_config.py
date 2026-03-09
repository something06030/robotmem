"""测试 config.py — 配置管理"""

import json
import pytest

from robotmem.config import Config, load_config, save_config


class TestConfig:

    def test_defaults(self):
        c = Config()
        assert c.embed_backend == "onnx"
        assert c.collection == "default"
        assert c.default_confidence == 0.9
        assert c.effective_embedding_dim == 384  # onnx default

    def test_ollama_dim(self):
        c = Config(embed_backend="ollama")
        assert c.effective_embedding_dim == 768

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="embed_backend"):
            Config(embed_backend="invalid")

    def test_invalid_api(self):
        with pytest.raises(ValueError, match="embed_api"):
            Config(embed_api="invalid")

    def test_db_path_default(self):
        c = Config()
        assert "memory.db" in c.db_path

    def test_db_path_custom(self):
        c = Config(db_path="/tmp/custom.db")
        assert str(c.db_path_resolved) == "/tmp/custom.db"

    def test_default_collection_property(self):
        c = Config(collection="robot-arm")
        assert c.default_collection == "robot-arm"


class TestLoadSaveConfig:

    def test_load_default(self, tmp_path, monkeypatch):
        """无配置文件时用默认值"""
        monkeypatch.setattr("robotmem.config.CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr("robotmem.config.ROBOTMEM_HOME", tmp_path)
        c = load_config()
        assert c.embed_backend == "onnx"

    def test_save_and_load(self, tmp_path, monkeypatch):
        """保存 → 读取往返"""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("robotmem.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("robotmem.config.ROBOTMEM_HOME", tmp_path)

        c = Config(collection="my-robot")
        save_config(c, config_file)

        # 验证文件内容
        with open(config_file) as f:
            data = json.load(f)
        assert data["collection"] == "my-robot"

        # 重新加载
        c2 = load_config()
        assert c2.collection == "my-robot"
