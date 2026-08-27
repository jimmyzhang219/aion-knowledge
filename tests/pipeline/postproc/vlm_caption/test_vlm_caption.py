from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher
from aion_knowledge.pipeline.postproc.vlm_caption.processor import VLMCaptionModule


def test_module_discovery():
    """验证模块可被调度器发现。"""
    dispatcher = PostProcDispatcher({"vlm_caption": True})
    assert "vlm_caption" in dispatcher._modules
    mod = dispatcher._modules["vlm_caption"]
    assert mod.always_on is True
    assert "text" in mod.depends_on


def test_module_factory():
    """验证 module() 工厂函数。"""
    from aion_knowledge.pipeline.postproc.vlm_caption.processor import module
    mod = module()
    assert isinstance(mod, VLMCaptionModule)
