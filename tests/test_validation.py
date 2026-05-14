"""Tests for edge_train.validation."""

from edge_train.validation import check_size_constraint


class TestCheckSizeConstraint:
    def test_under_limit(self):
        assert check_size_constraint(5.0) is True

    def test_at_limit(self):
        assert check_size_constraint(10.0) is True

    def test_over_limit(self):
        assert check_size_constraint(10.1) is False

    def test_custom_limit(self):
        assert check_size_constraint(20.0, limit_mb=25.0) is True
        assert check_size_constraint(30.0, limit_mb=25.0) is False
