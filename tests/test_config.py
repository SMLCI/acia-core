"""Tests for the acia.config credentials store."""

import os
import sys

import pytest

from acia import config
from acia.config import resolve_storage_options


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Ensure the lru-cached config loader does not leak between tests."""
    config._load_config.cache_clear()
    yield
    config._load_config.cache_clear()


def _write_config(tmp_path, content: str, monkeypatch) -> str:
    path = tmp_path / "credentials.toml"
    path.write_text(content)
    monkeypatch.setenv("ACIA_CONFIG", str(path))
    return str(path)


def test_local_path_returns_explicit(monkeypatch, tmp_path):
    """Plain local paths get no credential lookup; explicit options pass through."""
    _write_config(tmp_path, '[smb."srv"]\nusername = "x"\n', monkeypatch)

    assert resolve_storage_options("/data/img.tif") == {}
    assert resolve_storage_options("/data/img.tif", {"a": 1}) == {"a": 1}
    # explicit local protocol also skips lookup
    assert resolve_storage_options("file:///data/img.tif") == {}


def test_plaintext_entry(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        '[smb."fileserver.lab"]\nusername = "jdoe"\npassword = "secret"\n',
        monkeypatch,
    )

    opts = resolve_storage_options("smb://fileserver.lab/data/img.tif")
    assert opts == {"username": "jdoe", "password": "secret"}


def test_env_indirection(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        '[smb."srv"]\nusername = "svc"\npassword_env = "ACIA_TEST_PW"\n',
        monkeypatch,
    )
    monkeypatch.setenv("ACIA_TEST_PW", "from-env")

    opts = resolve_storage_options("smb://srv/share/img.tif")
    assert opts == {"username": "svc", "password": "from-env"}


def test_env_indirection_missing_raises(monkeypatch, tmp_path):
    _write_config(
        tmp_path, '[smb."srv"]\npassword_env = "ACIA_DOES_NOT_EXIST"\n', monkeypatch
    )
    monkeypatch.delenv("ACIA_DOES_NOT_EXIST", raising=False)

    with pytest.raises(KeyError):
        resolve_storage_options("smb://srv/share/img.tif")


def test_keyring_entry(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        '[smb."srv"]\nusername = "jdoe"\nkeyring = true\n',
        monkeypatch,
    )

    captured = {}

    class FakeKeyring:
        @staticmethod
        def get_password(service, username):
            captured["service"] = service
            captured["username"] = username
            return "kr-secret"

    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring)

    opts = resolve_storage_options("smb://srv/share/img.tif")
    assert opts == {"username": "jdoe", "password": "kr-secret"}
    assert captured == {"service": "smb://srv", "username": "jdoe"}


def test_keyring_missing_entry_raises(monkeypatch, tmp_path):
    _write_config(
        tmp_path, '[smb."srv"]\nusername = "jdoe"\nkeyring = true\n', monkeypatch
    )

    class FakeKeyring:
        @staticmethod
        def get_password(service, username):
            return None

    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring)

    with pytest.raises(KeyError):
        resolve_storage_options("smb://srv/share/img.tif")


def test_explicit_overrides_config(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        '[smb."srv"]\nusername = "config-user"\npassword = "config-pw"\n',
        monkeypatch,
    )

    opts = resolve_storage_options(
        "smb://srv/share/img.tif", {"password": "explicit-pw"}
    )
    assert opts == {"username": "config-user", "password": "explicit-pw"}


def test_protocol_defaults_merge_with_host(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        '[smb]\ndomain = "LAB"\n\n[smb."srv"]\nusername = "jdoe"\n',
        monkeypatch,
    )

    opts = resolve_storage_options("smb://srv/share/img.tif")
    assert opts == {"domain": "LAB", "username": "jdoe"}


def test_missing_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ACIA_CONFIG", str(tmp_path / "does-not-exist.toml"))
    assert resolve_storage_options("smb://srv/share/img.tif") == {}


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX-only")
def test_loose_permissions_warn(monkeypatch, tmp_path):
    path = _write_config(tmp_path, '[smb."srv"]\nusername = "jdoe"\n', monkeypatch)
    os.chmod(path, 0o644)  # group/other readable

    with pytest.warns(UserWarning, match="accessible by other users"):
        resolve_storage_options("smb://srv/share/img.tif")
