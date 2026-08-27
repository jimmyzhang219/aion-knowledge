from aion_knowledge.common.config import Settings


def test_vlm_config_defaults():
    s = Settings(_env_file=None)
    assert hasattr(s, "vlm_model")
    assert s.vlm_model == "qwen-vl-plus"
    # 未设置 provider 时应为 None（降级到 llm_provider）
    assert s.vlm_provider is None
    assert s.vlm_request_timeout == 60
