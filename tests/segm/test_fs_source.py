"""Tests for the fsspec-based image source plumbing (local + remote backends)."""

import fsspec
import numpy as np
import pytest
import tifffile

from acia.segm.local import (
    LocalSequenceSource,
    SambaSequenceSource,
    list_sequence_sources,
)


def _make_stack(num_frames=3, h=8, w=8):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(num_frames, h, w), dtype=np.uint8)


def _make_uint16_stack(num_frames=3, h=8, w=8):
    rng = np.random.default_rng(0)
    return rng.integers(0, 65535, size=(num_frames, h, w), dtype=np.uint16)


def test_local_path_backward_compat(tmp_path):
    """Reading a local TIFF through fsspec matches the original behavior."""
    stack = _make_stack()
    path = tmp_path / "stack.tif"
    tifffile.imwrite(str(path), stack)

    src = LocalSequenceSource(str(path))

    assert src.size_t == len(stack)
    assert src.num_channels == 3  # prepare_image makes 2D frames artificially RGB
    frames = list(src)
    assert len(frames) == len(stack)
    assert frames[0].raw.shape == (8, 8, 3)


def test_local_path_to_channel(tmp_path):
    """to_channel() selects one of the artificially-RGB-repeated channels."""
    stack = _make_stack()
    path = tmp_path / "stack.tif"
    tifffile.imwrite(str(path), stack)

    src = LocalSequenceSource(str(path))
    single = src.to_channel(0)

    assert single.size_t == len(stack)
    frame = single.get_frame(0)
    assert frame.raw.shape == (8, 8)
    np.testing.assert_array_equal(frame.raw, src.get_frame(0).raw[..., 0])


def test_local_path_normalize_image_false_preserves_raw_dtype_and_channel(tmp_path):
    """normalize_image=False: get_frame/indexing return true dtype, (H, W, 1) shape.

    Regression test for a bug where get_frame() ignored self.normalize_image
    (unlike __iter__), and where the 2D->3-channel duplication was unconditional
    -- both are now gated by normalize_image.
    """
    stack = _make_uint16_stack()
    path = tmp_path / "stack16.tif"
    tifffile.imwrite(str(path), stack)

    src = LocalSequenceSource(str(path), normalize_image=False)

    frame = src.get_frame(0)
    assert frame.raw.dtype == np.uint16
    assert frame.raw.shape == (8, 8, 1)
    np.testing.assert_array_equal(frame.raw[..., 0], stack[0])

    # __getitem__ (backed by get_frame) matches
    indexed = src[0]
    assert indexed.raw.dtype == np.uint16
    assert indexed.raw.shape == (8, 8, 1)

    assert src.num_channels == 1


def test_memory_backend_end_to_end():
    """The non-local fsspec path (same code SAMBA uses) reads a TIFF correctly."""
    stack = _make_stack(num_frames=2)
    with fsspec.open("memory://test.tif", mode="wb") as f:
        tifffile.imwrite(f, stack)

    src = LocalSequenceSource("memory://test.tif")

    assert src.size_t == 2
    frame = src.get_frame(0)
    assert frame.raw.shape == (8, 8, 3)


def test_samba_url_and_options():
    """SambaSequenceSource builds the smb URL and forwards explicit credentials."""
    src = SambaSequenceSource(
        host="srv",
        share="data",
        path="sub/img.tif",
        username="u",
        password="p",
    )

    assert src.filename == "smb://srv/data/sub/img.tif"
    assert src.storage_options == {
        "host": "srv",
        "username": "u",
        "password": "p",
    }


def test_samba_url_strips_leading_slash():
    src = SambaSequenceSource(host="srv", share="data", path="/abs/img.tif")

    assert src.filename == "smb://srv/data/abs/img.tif"
    # credentials omitted -> only host kept, rest resolved from config at read time
    assert src.storage_options == {"host": "srv"}


def test_samba_from_url():
    src = SambaSequenceSource.from_url("smb://fileserver.lab/data/exp/img.tif")

    assert isinstance(src, SambaSequenceSource)
    assert src.filename == "smb://fileserver.lab/data/exp/img.tif"
    # nothing but the host -> credentials resolved from config at read time
    assert src.storage_options == {"host": "fileserver.lab"}


def test_samba_from_url_with_embedded_credentials():
    src = SambaSequenceSource.from_url("smb://user:pass@srv:4455/data/sub/img.tif")

    assert src.filename == "smb://srv/data/sub/img.tif"
    assert src.storage_options == {
        "host": "srv",
        "username": "user",
        "password": "pass",
        "port": 4455,
    }


def test_samba_from_url_rejects_non_smb():
    with pytest.raises(ValueError, match="smb://"):
        SambaSequenceSource.from_url("s3://bucket/img.tif")


def test_localsequencesource_accepts_smb_url():
    """A plain smb:// URL works directly on LocalSequenceSource too."""
    src = LocalSequenceSource("smb://srv/data/img.tif")
    assert src.filename == "smb://srv/data/img.tif"


def test_list_sequence_sources_local(tmp_path):
    """Discover all matching stacks in a local folder, sorted, readable."""
    for name in ("b.tif", "a.tif"):
        tifffile.imwrite(str(tmp_path / name), _make_stack(num_frames=2))
    (tmp_path / "notes.txt").write_text("ignore me")

    sources = list_sequence_sources(str(tmp_path), pattern="*.tif")

    assert len(sources) == 2
    assert all(isinstance(s, LocalSequenceSource) for s in sources)
    # sorted by path -> a before b
    assert [s.filename.split("/")[-1] for s in sources] == ["a.tif", "b.tif"]
    # and they actually read
    assert sources[0].size_t == 2


def test_list_sequence_sources_remote_builds_urls():
    """For a remote backend, returned sources carry full URLs (with host)."""
    fs = fsspec.filesystem("memory")
    for name in ("exp/1.tif", "exp/2.tif"):
        with fs.open("/" + name, "wb") as f:
            tifffile.imwrite(f, _make_stack(num_frames=2))

    sources = list_sequence_sources("memory://exp", pattern="*.tif")

    assert [s.filename for s in sources] == [
        "memory:///exp/1.tif",
        "memory:///exp/2.tif",
    ]
    assert sources[0].size_t == 2

    fs.store.clear()  # avoid leaking memory:// state into other tests
