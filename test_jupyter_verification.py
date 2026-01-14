#!/usr/bin/env python
"""Test script to verify Jupyter notebook visualization functionality."""

import numpy as np
from acia.segm.local import THWCSequenceSource

def test_basic_functionality():
    """Test basic visualization functionality."""
    print("Creating test data...")
    # Create test data: 5 frames, 256x256, 3 channels
    test_data = np.random.randint(0, 255, (5, 256, 256, 3), dtype=np.uint8)

    print("Creating THWCSequenceSource...")
    src = THWCSequenceSource(test_data)

    print(f"Source properties:")
    print(f"  - Number of frames (size_t): {src.size_t}")
    print(f"  - Number of channels: {src.num_channels}")
    print(f"  - Has _repr_html_: {hasattr(src, '_repr_html_')}")

    # Try calling _repr_html_ in non-Jupyter environment (should return None)
    print("\nTesting _repr_html_ in non-Jupyter environment...")
    result = src._repr_html_()
    if result is None:
        print("  ✓ Correctly returns None in non-Jupyter environment")
    else:
        print(f"  ✗ Unexpected result: {result}")
        return False

    print("\n✅ All basic checks passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
