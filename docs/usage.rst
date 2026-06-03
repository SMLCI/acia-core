=====
Usage
=====

To use AutomatedCellularImageAnalysis in a project::

    import acia


Remote image sources (SMB / SAMBA, S3, ...)
===========================================

Image sequences are read through `fsspec <https://filesystem-spec.readthedocs.io>`_,
so the same source classes work for local files and any fsspec-supported backend
(SMB/SAMBA shares, S3, HTTP, FTP, ...). For local files nothing changes::

    from acia.segm.local import LocalSequenceSource

    src = LocalSequenceSource("/data/experiments/pos1.tif")

To read from a SAMBA share, either pass a full ``smb://`` URL or use the
``SambaSequenceSource`` convenience class::

    from acia.segm.local import LocalSequenceSource, SambaSequenceSource

    # full URL (works on LocalSequenceSource too)
    src = LocalSequenceSource("smb://fileserver.lab/data/exp/pos1.tif")

    # convenience class from a URL ...
    src = SambaSequenceSource.from_url("smb://fileserver.lab/data/exp/pos1.tif")

    # ... or from explicit parts
    src = SambaSequenceSource(host="fileserver.lab", share="data",
                              path="exp/pos1.tif")

The SMB backend requires the optional ``remote`` extra::

    pip install acia[remote]


Storing credentials
-------------------

Rather than putting secrets in every script, credentials are stored once in a
per-user config file and resolved automatically by **host**. The file lives in
the OS-standard config directory (override with the ``ACIA_CONFIG`` environment
variable):

* Linux:   ``~/.config/acia/credentials.toml``
* macOS:   ``~/Library/Application Support/acia/credentials.toml``
* Windows: ``%APPDATA%\acia\credentials.toml``

It is TOML, keyed by ``[<protocol>."<host>"]`` (with an optional ``[<protocol>]``
table for protocol-wide defaults). Each entry provides one of three secret
mechanisms (precedence: keyring → env-var → plaintext):

.. code-block:: toml

    # OS keyring (most secure): password from the OS secret store,
    #   service "smb://<host>", username = the `username` value
    [smb."fileserver.lab"]
    username = "jdoe"
    domain   = "LAB"
    keyring  = true

    # env-var indirection: any *_env key is read from the environment
    [smb."other-host"]
    username     = "svc"
    password_env = "ACIA_OTHER_PW"

    # plaintext (discouraged; acia warns if the file is group/other-readable)
    [s3."my-bucket"]
    key    = "AKIA..."
    secret = "..."

With an entry in place, no secrets are needed in code -- they are matched by the
host in the URL::

    src = SambaSequenceSource.from_url("smb://fileserver.lab/data/exp/pos1.tif")

Values passed explicitly (as keyword arguments, in the URL as
``smb://user:pass@host/...``, or via ``storage_options``) always override the
config file.


Discovering all stacks in a folder
----------------------------------

``list_sequence_sources`` returns one source per matching file in a folder. It
uses the same fsspec + credential layer, so it works for a local directory or a
remote share::

    from acia.segm.local import list_sequence_sources

    # local
    for src in list_sequence_sources("/data/experiments", pattern="*.tif"):
        analyse(src)

    # SMB share -- credentials resolved by host, none in code
    sources = list_sequence_sources("smb://fileserver.lab/data/exp", pattern="*.tif")

Use ``recursive=True`` to walk sub-folders, ``storage_options=...`` to override
credentials, and pass extra keyword arguments (``normalize_image``, ``luts``,
``channel_index``) through to each source.
