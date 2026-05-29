"""Streaming top-k Triton kernel is parked — see module docstring."""

import pytest


@pytest.mark.skip(reason="Streaming top-k kernel parked; see module docstring.")
def test_streaming_topk_matches_torch_topk():
    pass
