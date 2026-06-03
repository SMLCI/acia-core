"""Filesystem credentials store for image sources.

ACIA reads image sequences through fsspec, which means a single uniform
interface works for local files, SMB/SAMBA shares, S3, HTTP, FTP, ... Each remote
backend usually needs credentials (``storage_options`` in fsspec terms). To avoid
sprinkling secrets across user scripts, those credentials can be stored once in a
per-user config file and are resolved automatically by ``resolve_storage_options``.

Config file
-----------
Location: ``<user_config_dir>/acia/credentials.toml`` where ``<user_config_dir>``
follows the OS convention (via :mod:`platformdirs`):

* Linux:   ``~/.config/acia/``
* macOS:   ``~/Library/Application Support/acia/``
* Windows: ``%APPDATA%\\acia\\``

The path can be overridden with the ``ACIA_CONFIG`` environment variable.

The file is TOML, keyed by ``[<protocol>."<host>"]`` with an optional
``[<protocol>]`` table holding protocol-wide defaults. Every key in a table is
forwarded to fsspec as a ``storage_options`` entry, *except* the secret-resolution
keys below, which are evaluated first (precedence: keyring -> env -> plaintext):

.. code-block:: toml

    # OS keyring (most secure): password fetched from the OS secret store,
    #   service = "smb://<host>", username = <username>
    [smb."fileserver.lab"]
    username = "jdoe"
    domain   = "LAB"
    keyring  = true

    # env-var indirection: any *_env key resolves from the environment and is
    #   stored under the stripped key name (password_env -> password)
    [smb."other-host"]
    username     = "svc"
    password_env = "ACIA_OTHER_PW"

    # plaintext (discouraged; file perms are checked, see below)
    [s3."my-bucket"]
    key    = "AKIA..."
    secret = "..."
"""

from __future__ import annotations

import os
import stat
import sys
import warnings
from functools import cache
from urllib.parse import urlsplit

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


CONFIG_FILENAME = "credentials.toml"


def _config_path() -> str:
    """Return the path to the credentials config file.

    Honors the ``ACIA_CONFIG`` environment variable; otherwise uses the
    OS-standard per-user config directory.
    """
    override = os.environ.get("ACIA_CONFIG")
    if override:
        return override

    import platformdirs

    return os.path.join(platformdirs.user_config_dir("acia"), CONFIG_FILENAME)


def _warn_on_loose_permissions(path: str) -> None:
    """Warn (SSH-style) if the config file is readable by group/other (POSIX)."""
    if os.name != "posix":
        return
    mode = os.stat(path).st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        warnings.warn(
            f"acia credentials file {path!r} is accessible by other users. "
            f"Consider restricting it with `chmod 600 {path}`.",
            stacklevel=2,
        )


@cache
def _load_config(path: str, _mtime: float) -> dict:
    """Load and parse the TOML config file (cached by path + mtime).

    ``_mtime`` is part of the cache key so edits to the file are picked up
    without restarting the process.
    """
    _warn_on_loose_permissions(path)
    with open(path, "rb") as f:
        config: dict = tomllib.load(f)
    return config


def _read_config() -> dict:
    """Load the config file, returning an empty dict if it does not exist."""
    path = _config_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    return _load_config(path, mtime)


def _resolve_secrets(entry: dict, *, protocol: str, host: str | None) -> dict:
    """Resolve secret-indirection keys in a single config entry.

    Returns a new dict of fsspec ``storage_options`` with:

    * ``keyring = true`` -> ``password`` fetched from the OS keyring
      (service ``"<protocol>://<host>"``, username from the ``username`` key),
    * ``*_env`` keys -> looked up in the environment, stored under the stripped
      key name,
    * all other keys passed through verbatim.
    """
    options: dict = {}
    use_keyring = False

    for key, value in entry.items():
        if key == "keyring":
            use_keyring = bool(value)
            continue
        if key.endswith("_env"):
            target = key[: -len("_env")]
            env_value = os.environ.get(value)
            if env_value is None:
                raise KeyError(
                    f"acia credentials: environment variable {value!r} (referenced "
                    f"by {key!r} for {protocol}://{host}) is not set."
                )
            options[target] = env_value
            continue
        options[key] = value

    if use_keyring:
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "acia credentials entry requests `keyring = true` but the `keyring` "
                "package is not installed. Install it with `pip install acia[remote]` "
                "or provide the secret via an environment variable or plaintext."
            ) from exc

        service = f"{protocol}://{host}" if host else protocol
        username = options.get("username")
        password = keyring.get_password(service, username) if username else None
        if password is None:
            raise KeyError(
                f"acia credentials: no keyring entry found for service "
                f"{service!r} and username {username!r}."
            )
        options["password"] = password

    return options


def resolve_storage_options(url: str, explicit: dict | None = None) -> dict:
    """Resolve fsspec ``storage_options`` for ``url`` from the credentials config.

    Looks up the ``[<protocol>."<host>"]`` entry (falling back to a
    ``[<protocol>]`` default table) for the URL's protocol and host, resolves any
    secret-indirection keys, and merges ``explicit`` options on top (explicit
    always wins).

    For local paths or URLs without a remote protocol, returns the ``explicit``
    options unchanged (an empty dict if none given), so existing local usage is
    unaffected.
    """
    explicit = dict(explicit) if explicit else {}

    protocol, _, _ = url.partition("://")
    if not _ or protocol in ("file", "local"):
        # plain local path or explicit local protocol -> no credential lookup
        return explicit

    host = urlsplit(url).hostname

    config = _read_config()
    proto_section = config.get(protocol, {})

    options: dict = {}
    # protocol-wide defaults: keys that are not themselves host tables
    defaults = {k: v for k, v in proto_section.items() if not isinstance(v, dict)}
    if defaults:
        options.update(_resolve_secrets(defaults, protocol=protocol, host=host))
    # host-specific entry
    if host is not None and isinstance(proto_section.get(host), dict):
        options.update(
            _resolve_secrets(proto_section[host], protocol=protocol, host=host)
        )

    options.update(explicit)
    return options
