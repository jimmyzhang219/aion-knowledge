"""测试图片解析器。"""
import base64

from aion_knowledge.pipeline.parser.image_resolver import (
    ImageResolver,
    is_icon_image,
    resolve_images,
)


class TestIsIconImage:
    def test_small_dimensions_is_icon(self):
        """10x10 图片应被视为图标。"""
        data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG header
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 pixel
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # ...
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT
            0x54, 0x08, 0xD7, 0x63, 0x60, 0x60, 0x00, 0x00,  # ...
            0x00, 0x02, 0x00, 0x01, 0xE5, 0x27, 0xDE, 0xFC,  # ...
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44,  # IEND
            0xAE, 0x42, 0x60, 0x82,
        ])
        assert is_icon_image(data) is True

    def test_large_dimensions_not_icon(self):
        """200x200 图片不应被视为图标。"""
        import io

        from PIL import Image
        img = Image.new("RGB", (200, 200), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        assert is_icon_image(buf.getvalue()) is False

    def test_undecodable_small_bytes_is_icon(self):
        """无法解码的极小数据应视为图标（回退到字节大小阈值）。"""
        data = b"x" * 100  # 100 bytes < 512
        assert is_icon_image(data) is True

    def test_undecodable_large_bytes_not_icon(self):
        data = b"x" * 600  # 600 bytes > 512
        assert is_icon_image(data) is False


class TestImageResolverBasic:
    def test_resolve_images_basic(self):
        """基础 markdown 图片引用解析。"""
        content = "前文![测试](https://example.com/img.png)后文"
        updated, images = resolve_images(content)
        # 远程图片在任务 4 中集成 SSRF 后才解析，当前保留原始引用
        assert "前文" in updated
        assert "后文" in updated
        assert "https://example.com/img.png" in updated

    def test_resolve_images_data_uri(self):
        """data URI 图片引用被正确解析。"""
        # 构造一个 100x100 的 PNG（避免被 is_icon_image 过滤）
        import io

        from PIL import Image
        img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        resolver = ImageResolver()
        content = f"![img](data:image/png;base64,{b64})"
        updated, images = resolver.resolve_images(content)
        assert "images/" in updated
        assert len(images) == 1

    def test_existing_images_preserved(self):
        """传入 existing_images 时，已有图片不被重复处理。"""
        existing = {"images/old.png": "base64data"}
        resolver = ImageResolver()
        updated, images = resolver.resolve_images("无图片", existing)
        assert "images/old.png" in images
        assert len(images) == 1

    def test_data_uri_icon_filtered(self):
        """data URI 中的 1x1 图标应被过滤（返回空 alt）。"""
        b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        content = f"![icon](data:image/png;base64,{b64})"
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        assert len(images) == 0
        # 图标被过滤，引用被移除
        assert "![icon](" not in updated


class TestHTMLImageResolution:
    def test_html_data_uri_img(self):
        """HTML <img> 标签中的 data:URI 应被解析。"""
        import io

        from PIL import Image
        img = Image.new("RGB", (100, 100), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content = f'前文<img src="data:image/png;base64,{b64}" alt="test">后文'
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        assert "images/" in updated
        assert len(images) == 1

    def test_html_relative_src_img(self, tmp_path):
        """HTML <img> 中的相对路径应被解析。"""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="blue")
        img_path = tmp_path / "photo.png"
        img.save(str(img_path), format="PNG")
        content = f'<img src="{img_path}" alt="photo">'
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content, {})
        assert len(images) == 1

    def test_html_img_http_src_preserved(self):
        """HTML <img> 中的 http(s) src 被保留（远程图片由后续任务处理）。"""
        content = '<img src="https://cdn.example.com/photo.jpg" alt="photo">'
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        # 远程图片在 _resolve_remote_images 中处理，此处保留原始标签
        assert '<img' in updated or "https://cdn.example.com/photo.jpg" in updated

    def test_html_img_icon_filtered(self):
        """被 is_icon_image 过滤的 HTML img 引用应被移除。"""
        b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        content = f'<img src="data:image/png;base64,{b64}" alt="icon">'
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        # 1x1 图标应被过滤，标签被移除
        assert "<img" not in updated
        assert len(images) == 0


class TestRemoteImageSSRF:
    def test_remote_image_ssrf_blocked_preserved(self):
        """SSRF 拦截的远程图片保留原始引用。"""
        content = "![img](http://10.0.0.1/secret.png)"
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        assert "http://10.0.0.1/secret.png" in updated
        assert len(images) == 0

    def test_remote_image_success(self, httpserver, monkeypatch):
        """可访问的远程图片成功解析（需白名单 localhost）。"""
        from aion_knowledge.infrastructure.security import reset_ssrf_whitelist
        monkeypatch.setenv("SSRF_WHITELIST", "localhost,127.0.0.1")
        reset_ssrf_whitelist()

        import io

        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        httpserver.expect_request("/img.png").respond_with_data(
            data, content_type="image/png"
        )
        content = f"![img]({httpserver.url_for('/img.png')})"
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        assert "images/" in updated
        assert len(images) == 1
        # 清理 whitelist 避免影响后续测试
        monkeypatch.delenv("SSRF_WHITELIST", raising=False)
        reset_ssrf_whitelist()


class TestCountLimits:
    def test_max_remote_limit(self):
        """超过 max_remote 后被跳过。"""
        resolver = ImageResolver(max_remote=2)
        content = "![img](https://example.com/img.png)![img2](https://example.com/img2.png)"
        resolver._remote_count = 2
        updated, images = resolver.resolve_images(content)
        assert "example.com" in updated
        assert len(images) == 0

    def test_max_data_uri_limit(self):
        """超过 max_data_uri 的被跳过。"""
        import io

        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resolver = ImageResolver(max_data_uri=1)
        content = f"![1](data:image/png;base64,{b64})![2](data:image/png;base64,{b64})"
        updated, images = resolver.resolve_images(content)
        assert len(images) == 1

    def test_single_size_limit(self):
        """超过 max_single_size 的被跳过。"""
        resolver = ImageResolver(max_single_size=10)
        content = "![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==)"
        updated, images = resolver.resolve_images(content)
        assert len(images) == 0


class TestDedup:
    def test_same_url_dedup(self):
        """相同 URL 只解析一次。"""
        import io

        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        content = f"![a](data:image/png;base64,{b64})![b](data:image/png;base64,{b64})"
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        assert len(images) == 1

    def test_filename_dedup(self, tmp_path):
        """相同文件名的不同路径引用只上传一次。"""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        path1 = tmp_path / "images" / "foo.png"
        path1.parent.mkdir(parents=True)
        img.save(str(path1))
        path2 = tmp_path / "subdir" / "foo.png"
        path2.parent.mkdir(parents=True)
        img.save(str(path2))
        content = f"![a]({path1})![b]({path2})"
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content)
        assert len(images) == 1
