import json
from pathlib import Path

import pytest

from neurotarget.infinite_gateway import check_connection


def _configuration(tmp_path, origin="https://infinite.example/api"):
    path = tmp_path / "outputs/private/infinite_gateway.auth.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"api_base": origin, "cookies": {"gateway": "unit-test-only"}}))


def test_gateway_cookies_cannot_follow_a_redirect_or_cross_origin(tmp_path, monkeypatch):
    _configuration(tmp_path)
    monkeypatch.setenv("INFINITE_API_BASE", "https://other.example/api")
    monkeypatch.delenv("INFINITE_GATEWAY_CONFIG", raising=False)
    with pytest.raises(ValueError, match="authenticated HTTPS"):
        check_connection(tmp_path)


def test_read_only_gateway_check(monkeypatch, tmp_path):
    _configuration(tmp_path)
    monkeypatch.delenv("INFINITE_API_BASE", raising=False)
    monkeypatch.delenv("INFINITE_GATEWAY_CONFIG", raising=False)
    calls = []
    class Response:
        status_code = 200
        def json(self): return {"communities": [{"name": "biology"}]}
    def get(session, url, **kwargs):
        calls.append((url, kwargs))
        assert session.cookies.get("gateway") == "unit-test-only"
        return Response()
    monkeypatch.setattr("requests.Session.get", get)
    result = check_connection(tmp_path)
    assert result["gateway_authenticated"]
    assert result["publication_enabled"] is False
    assert len(calls) == 1 and calls[0][1]["allow_redirects"] is False


def test_insecure_gateway_origin_is_rejected(monkeypatch, tmp_path):
    _configuration(tmp_path, "http://infinite.example/api")
    monkeypatch.delenv("INFINITE_API_BASE", raising=False)
    monkeypatch.delenv("INFINITE_GATEWAY_CONFIG", raising=False)
    with pytest.raises(ValueError, match="HTTPS"):
        check_connection(tmp_path)
