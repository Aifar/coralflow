"""Tests for edge_train.cli.cost._estimate_hours."""

import pytest
from edge_train.cli.cost import _estimate_hours


class TestEstimateHours:
    def test_text_small(self):
        h = _estimate_hours("text", 400)
        assert h == pytest.approx(0.064, rel=0.1)

    def test_image(self):
        h = _estimate_hours("image", 200)
        assert h == pytest.approx(0.048, rel=0.1)

    def test_table(self):
        h = _estimate_hours("table", 5000)
        assert h == pytest.approx(0.5, rel=0.1)

    def test_zero_samples(self):
        assert _estimate_hours("text", 0) == 0.0
