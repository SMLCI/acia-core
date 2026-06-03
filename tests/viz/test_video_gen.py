"""Module for testing visualization module"""

import os
import unittest
from pathlib import Path

import numpy as np
import pytest

from acia.segm.local import InMemorySequenceSource
from acia.segm.output import renderVideo
from acia.viz import VideoExporter, VideoExporter2

# encodes real video files via moviepy/ffmpeg
pytestmark = pytest.mark.integration


class TestVideoGen(unittest.TestCase):
    """Testing scalebar and time overlay"""

    def _test_encoding(self, video_file: Path, codec):
        """Random images to video"""

        images = np.random.randint(0, 255, size=(10, 200, 200, 3), dtype=np.uint8)

        with VideoExporter(str(video_file.absolute()), 3, codec=codec) as ve:
            for image in images:
                ve.write(image)

        # Check that video file exists
        self.assertTrue(video_file.exists)
        # Check that it has some content (>10 KB)
        self.assertTrue(os.path.getsize(video_file) > 10 * 1024)

    def _test_encoder(self, encoder):
        images = np.random.randint(0, 255, size=(20, 200, 200, 3), dtype=np.uint8)

        with encoder as ve:
            for image in images:
                ve.write(image)

        # Check that video file exists
        self.assertTrue(encoder.filename.exists)
        # Check that it has some content (>10 KB)
        self.assertTrue(os.path.getsize(encoder.filename) > 10 * 1024)

    def test_vp9(self):
        """Encode video in vp9"""
        self._test_encoding(Path("test.mp4"), codec="vp09")

    def test_avi(self):
        """Encode video in avi"""
        self._test_encoding(Path("test.avi"), codec="MJPG")

    def test_avi_fail(self):
        """Encode with wrong codec parameter"""
        with self.assertRaises(TypeError):
            self._test_encoding(Path("test.avi"), codec="JPG")

    def test_new_vp9(self):
        self._test_encoder(VideoExporter2.default_vp9(Path("testvp9.mp4"), 3))

    def test_new_h264(self):
        self._test_encoder(VideoExporter2.default_h264(Path("testh264.mp4"), 3))

    def test_new_h265(self):
        self._test_encoder(VideoExporter2.default_h265(Path("testh265.mp4"), 3))


class TestVideoRendering(unittest.TestCase):
    """Testing video rendering with different codecs"""

    def _test_rendering(self, filepath: Path, codec: str, framerate: int):
        """Test video rendering with specific parameters

        Args:
            filepath (Path): video file path
            codec (str): codec name
            framerate (int): video framerate (fps)
        """

        # random images
        images = np.random.randint(0, 255, size=(20, 200, 200, 3), dtype=np.uint8)
        source = InMemorySequenceSource(images)

        # remove existing files
        filepath = Path(filepath)
        if filepath.exists():
            os.remove(filepath)

        # render videos
        renderVideo(source, filename=filepath, framerate=framerate, codec=codec)

        # Check that video file exists
        self.assertTrue(filepath.exists)
        # Check that it has some content (>10 KB)
        self.assertTrue(os.path.getsize(filepath) > 10 * 1024)

    def test_vp9(self):
        """Encode video in vp9"""
        self._test_rendering("test.mp4", codec="vp09", framerate=10)

    def test_avi(self):
        """Encode video in mjpg"""
        self._test_rendering("test.avi", codec="MJPG", framerate=10)

    def test_avi_fail(self):
        """Encode with wrong codec parameter"""
        with self.assertRaises(ValueError):
            self._test_rendering("test.avi", codec="JPG", framerate=10)

    def test_h264(self):
        """Encode video in h264"""
        self._test_rendering("testh264.mp4", codec="h264", framerate=10)

    def test_new_h265(self):
        """Encode video in h265"""
        self._test_rendering("testh265.mp4", codec="h265", framerate=10)


if __name__ == "__main__":
    unittest.main()
