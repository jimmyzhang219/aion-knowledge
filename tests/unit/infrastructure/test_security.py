"""测试 SSRF 安全验证模块。"""
import pytest

from aion_knowledge.infrastructure.security import (
    SSRFError,
    validate_url_for_ssrf,
)


class TestValidateURLForSSRF:
    def test_blocks_private_ipv4(self):
        """10.x.x.x 应被拦截。"""
        with pytest.raises(SSRFError, match="restricted"):
            validate_url_for_ssrf("http://10.0.0.1/image.png")

    def test_blocks_loopback_ipv4(self):
        """127.x.x.x 应被拦截。"""
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://127.0.0.1:9000/img.png")

    def test_blocks_loopback_ipv6(self):
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://[::1]:8080/img.png")

    def test_blocks_localhost_hostname(self):
        with pytest.raises(SSRFError, match="localhost"):
            validate_url_for_ssrf("http://localhost/image.png")

    def test_blocks_metadata_hostname(self):
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://metadata.google.internal/computeMetadata/v1/")

    def test_blocks_link_local(self):
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://169.254.169.254/latest/meta-data/")

    def test_blocks_internal_service_port(self):
        """PostgreSQL 默认端口应被拦截。"""
        with pytest.raises(SSRFError, match="port"):
            validate_url_for_ssrf("http://example.com:5432/")

    def test_allows_public_domain(self):
        """公开域名应放行（DNS 可解析时）。"""
        try:
            validate_url_for_ssrf("https://cdn.example.com/photo.jpg")
        except SSRFError as e:
            if "DNS" in str(e):
                pytest.skip(f"DNS 不可达: {e}")
            raise

    def test_blocks_reserved_host_suffix(self):
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://internal-server.internal/img.png")

    def test_rejects_non_http_scheme(self):
        with pytest.raises(SSRFError, match="scheme"):
            validate_url_for_ssrf("ftp://example.com/file")

    def test_rejects_empty_url(self):
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("")

    def test_blocks_cgnat_range(self):
        """100.64.0.0/10 应被拦截。"""
        with pytest.raises(SSRFError):
            validate_url_for_ssrf("http://100.64.0.1/img.png")

    def test_whitelist_skips_dns_failure(self, monkeypatch):
        """白名单中的主机即使 DNS 不可达也放行（跳过 DNS 校验）。"""
        monkeypatch.setenv("SSRF_WHITELIST", "*.trusted-corp.com")
        import importlib

        import aion_knowledge.infrastructure.security as sec
        importlib.reload(sec)
        # .com 不在受限后缀，白名单跳过 DNS 校验
        sec.validate_url_for_ssrf("http://test.trusted-corp.com/img.png")

    def test_non_whitelist_dns_failure(self, monkeypatch):
        """非白名单且 DNS 不可达的主机应被拦截。"""
        # 清除环境变量确保无 whitelist
        monkeypatch.delenv("SSRF_WHITELIST", raising=False)
        import importlib

        import aion_knowledge.infrastructure.security as sec
        importlib.reload(sec)
        with pytest.raises(sec.SSRFError, match="DNS"):
            sec.validate_url_for_ssrf("http://test.trusted-corp.com/img.png")


class TestSSRFSafeHTTPClient:
    def test_redirect_to_internal_blocked(self, httpserver, monkeypatch):
        """重定向到内网地址应被拦截。"""
        monkeypatch.setenv("SSRF_WHITELIST", "localhost,127.0.0.1")
        import importlib

        import aion_knowledge.infrastructure.security as sec
        importlib.reload(sec)
        httpserver.expect_request("/redirect").respond_with_data(
            "", status=302, headers={"location": "http://10.0.0.1/secret"}
        )
        client = sec.SSRFSafeHTTPClient()
        with pytest.raises(sec.SSRFError):
            client.get(httpserver.url_for("/redirect"))

    def test_normal_get_succeeds(self, httpserver, monkeypatch):
        """正常请求应成功。"""
        monkeypatch.setenv("SSRF_WHITELIST", "localhost,127.0.0.1")
        import importlib

        import aion_knowledge.infrastructure.security as sec
        importlib.reload(sec)
        httpserver.expect_request("/img.png").respond_with_data(b"fake-image")
        client = sec.SSRFSafeHTTPClient()
        resp = client.get(httpserver.url_for("/img.png"))
        assert resp.status_code == 200
        assert resp.content == b"fake-image"
