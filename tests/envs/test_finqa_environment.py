# [MODULE DOC]
# Tests for the FinQA environment: reward matching via compute_reward,
# parse_number and extract_boxed_answer (no data needed; a prediction matches
# when both a relative tolerance, default 1%, and an absolute difference
# limit, default 1.0, are satisfied), plus data-dependent tool and environment
# tests. Prepends the repo root to sys.path.
r"""
Tests for the FinQA environment.

Reward matching tests (no data required) cover:
1. LaTeX escaped percentages (\%)
2. Decimal precision matching within tolerance
3. Ratios and small numbers
4. Regular numbers and edge cases

Integration tests (require data) cover:
- Tool implementations (get_descriptions, get_table_info, sql_query)
- Environment logic (reset, step, state)

Run from OpenEnv repo root:
    python -m pytest tests/envs/test_finqa_environment.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Add repo root to path so envs/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.finqa_env.server.rewards import (
    compute_reward,
    extract_boxed_answer,
    parse_number,
)


# ---------------------------------------------------------------------------
# Reward matching tests (no data dependency)
# ---------------------------------------------------------------------------

# Reward matching uses AND logic: both relative tolerance (1%) AND absolute difference (1.0) must pass


class TestRewards:
    # [CLASS DOC]
    # Basic compute_reward and parsing behaviour.
    """Test reward computation logic."""

    def test_exact_match(self):
        # [FUNCTION DOC]
        # Identical numbers score 1.0.
        assert compute_reward("6.118", "6.118") == 1.0

    def test_boxed_format(self):
        # [FUNCTION DOC]
        # \boxed{} on either side is unwrapped.
        assert compute_reward("6.118", r"\boxed{6.118}") == 1.0
        assert compute_reward(r"\boxed{6.118}", "6.118") == 1.0

    def test_tolerance(self):
        # [FUNCTION DOC]
        # Values within tolerance score 1.0.
        # Within 1% tolerance
        assert compute_reward("6.12", "6.118") == 1.0
        assert compute_reward("6.1", "6.118") == 1.0

    def test_incorrect(self):
        # [FUNCTION DOC]
        # Values outside tolerance score 0.0.
        assert compute_reward("5.0", "6.118") == 0.0
        assert compute_reward("100", "6.118") == 0.0

    def test_parse_number(self):
        # [FUNCTION DOC]
        # Handles decimals, thousands separators, percentages (as fractions) and fractions.
        assert parse_number("6.118") == 6.118
        assert parse_number("1,234.56") == 1234.56
        assert parse_number("20%") == 0.2
        assert parse_number("1/2") == 0.5

    def test_extract_boxed(self):
        # [FUNCTION DOC]
        # Extracts boxed content or returns None.
        assert extract_boxed_answer(r"\boxed{6.118}") == "6.118"
        assert extract_boxed_answer("no boxed here") is None


# finqa labels exist in "\boxed{...}" format, e.g. "\boxed{6.280\%}", so tests below are centered around this format
class TestLatexPercentages:
    # [CLASS DOC]
    # Ground truths with LaTeX-escaped percent signs.
    """Test LaTeX escaped percentage signs in ground truth."""

    def test_latex_escaped_percentage_exact_match(self):
        # [FUNCTION DOC]
        # Plain, escaped and boxed percentages all match.
        """Test exact match with LaTeX escaped %."""
        assert compute_reward("6.280%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"6.280\%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"\boxed{6.280\%}", r"\boxed{6.280\%}") == 1.0

    def test_latex_escaped_percentage_within_tolerance(self):
        # [FUNCTION DOC]
        # Lower precision and fractional equivalents match.
        """Test matching within decimal tolerance."""
        assert compute_reward("6.28%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"\boxed{6.28%}", r"\boxed{6.280\%}") == 1.0
        assert compute_reward("0.0628", r"\boxed{6.280\%}") == 1.0

    def test_latex_percentage_with_parentheses(self):
        # [FUNCTION DOC]
        # \( \) wrappers are stripped.
        """Test LaTeX format with parentheses wrapper."""
        assert compute_reward("-0.606%", r"\(\boxed{-0.606\%}\)") == 1.0

    def test_latex_dollar_signs(self):
        # [FUNCTION DOC]
        # $ wrappers are stripped.
        """Test LaTeX format with dollar sign wrappers."""
        assert compute_reward("57.73", r"$\boxed{57.730}$") == 1.0


class TestDecimalPrecisionMatching:
    # [CLASS DOC]
    # Percentage precision tolerance cases.
    """Test decimal precision matching within tolerance for percentages."""

    def test_percentage_1_decimal_point_diff(self):
        # [FUNCTION DOC]
        # 0.01 point difference matches.
        """6.29% vs 6.28% should match (0.01 percentage point)."""
        assert compute_reward("6.29%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"\boxed{6.29\%}", r"\boxed{6.280\%}") == 1.0
        assert compute_reward("0.0629", r"\boxed{6.280\%}") == 1.0

    def test_percentage_2_decimal_points_diff(self):
        # [FUNCTION DOC]
        # 0.02 point difference matches.
        """6.30% vs 6.28% should match (0.02 percentage point)."""
        assert compute_reward("6.30%", r"\boxed{6.280\%}") == 1.0

    def test_percentage_large_diff_should_fail(self):
        # [FUNCTION DOC]
        # 0.72 point difference fails.
        """7.00% vs 6.28% should NOT match (0.72 percentage point)."""
        assert compute_reward("7.00%", r"\boxed{6.280\%}") == 0.0

    def test_percentage_1_percent_point_diff_should_fail(self):
        # [FUNCTION DOC]
        # 1.0 point difference fails.
        """7.28% vs 6.28% should NOT match (1.0 percentage point)."""
        assert compute_reward("7.28%", r"\boxed{6.280\%}") == 0.0

    def test_percentage_precision_variation(self):
        # [FUNCTION DOC]
        # Different precisions of the same value match.
        """Test different precision levels."""
        assert compute_reward("25.14%", r"\boxed{25.144\%}") == 1.0
        assert compute_reward("25.144%", r"\boxed{25.144\%}") == 1.0
        assert compute_reward("25.1%", r"\boxed{25.144\%}") == 1.0

    def test_negative_percentage_precision(self):
        # [FUNCTION DOC]
        # Negative percentages match within tolerance.
        """Test negative percentages within tolerance."""
        assert compute_reward("-0.61%", r"\boxed{-0.606\%}") == 1.0
        assert compute_reward("-0.606%", r"\boxed{-0.606\%}") == 1.0


class TestRatiosAndSmallNumbers:
    # [CLASS DOC]
    # Ratio and small number cases.
    """Test ratio matching with appropriate decimal precision."""

    def test_ratio_exact_match(self):
        # [FUNCTION DOC]
        # Exact ratio matches.
        """Test exact ratio match."""
        assert compute_reward("0.232", r"\boxed{0.232}") == 1.0

    def test_ratio_1_decimal_diff(self):
        # [FUNCTION DOC]
        # 0.001 difference matches in several notations.
        """0.233 vs 0.232 should match (0.001 diff, within tolerance)."""
        assert compute_reward("0.233", r"\boxed{0.232}") == 1.0
        assert compute_reward(r"\boxed{0.233}", r"\boxed{0.232}") == 1.0
        assert compute_reward("233/1000", r"\boxed{0.232}") == 1.0
        assert compute_reward(r"$0.233$", r"\boxed{0.232}") == 1.0

    def test_ratio_3_decimal_diff_should_fail(self):
        # [FUNCTION DOC]
        # 0.003 difference (over 1% relative) fails.
        """0.235 vs 0.232 should NOT match (0.003 diff, exceeds relative tolerance)."""
        assert compute_reward("0.235", r"\boxed{0.232}") == 0.0

    def test_ratio_with_relative_tolerance(self):
        # [FUNCTION DOC]
        # Under 1% relative difference matches.
        """Test ratios within 1% relative tolerance."""
        # 0.321 vs 0.320 = 0.31% relative error
        assert compute_reward("0.321", r"\boxed{0.320}") == 1.0
        assert compute_reward("321/1000", r"\boxed{0.320}") == 1.0
        assert compute_reward(r"\boxed{0.321}", r"\boxed{0.320}") == 1.0

    def test_small_ratios(self):
        # [FUNCTION DOC]
        # Small ratios within tolerance match.
        """Test very small ratio values."""
        assert compute_reward("0.046", r"\boxed{0.046}") == 1.0
        assert compute_reward("0.0463", r"\boxed{0.046}") == 1.0


class TestRegularNumbers:
    # [CLASS DOC]
    # Integers and large values.
    """Test regular numbers and large values."""

    def test_negative_numbers(self):
        # [FUNCTION DOC]
        # Negative and accounting-parenthesis forms match ground truth with units text.
        """Test negative number matching."""
        assert compute_reward("-77", r"\boxed{-77} million") == 1.0
        assert compute_reward(r"\boxed{-77}", r"\boxed{-77} million") == 1.0
        assert compute_reward("(77)", r"\boxed{-77} million") == 1.0

    def test_large_numbers_with_relative_tolerance(self):
        # [FUNCTION DOC]
        # Both relative and absolute limits must pass.
        """Test large numbers must pass BOTH relative AND absolute thresholds."""
        # 1000 vs 1001 = 0.1% relative error, abs diff = 1.0, passes both
        assert compute_reward("1001", r"\boxed{1000}") == 1.0
        # 1000 vs 1009 = 0.9% relative error but abs diff = 9 > 1.0, fails
        assert compute_reward("1009", r"\boxed{1000}") == 0.0
        # 1000 vs 1011 = 1.1% relative error, should fail
        assert compute_reward("1011", r"\boxed{1000}") == 0.0

    def test_decimal_numbers(self):
        # [FUNCTION DOC]
        # Decimal exact match.
        """Test decimal number matching."""
        assert compute_reward("6.118", r"\boxed{6.118}") == 1.0

    def test_thousands_separators(self):
        # [FUNCTION DOC]
        # Thousands separators on either side are ignored.
        """Test numbers with thousand separators."""
        assert compute_reward("1,234.56", r"\boxed{1234.56}") == 1.0
        assert compute_reward("1234.56", r"\boxed{1,234.56}") == 1.0
        assert compute_reward(r"\boxed{1,234.56}", r"\boxed{1234.56}") == 1.0


class TestEdgeCases:
    # [CLASS DOC]
    # Edge cases.
    """Test edge cases and special scenarios."""

    def test_zero_values(self):
        # [FUNCTION DOC]
        # Zero matches zero.
        """Test zero value matching."""
        assert compute_reward("0", r"\boxed{0}") == 1.0
        assert compute_reward("0.0", r"\boxed{0}") == 1.0

    def test_percentage_points_notation(self):
        # [FUNCTION DOC]
        # "X%" matches a bare X ground truth (percentage-point fallback).
        """Test percentage points fallback: "4.5%" should match "4.500" (both mean 4.5 percentage points)."""
        # Walmart tax rate differential: "4.5%" vs "\boxed{4.500}" - should match
        assert compute_reward("4.5%", r"\boxed{4.500}") == 1.0
        # GM tax rate difference: "6.0%" vs "\boxed{6.000}" - should match
        assert compute_reward("6.0%", r"\boxed{6.000}") == 1.0
        # General case
        assert compute_reward("6.28%", r"\boxed{6.28}") == 1.0

    def test_fractions(self):
        # [FUNCTION DOC]
        # Fractions, decimals and percentages are interchangeable.
        """Test fraction matching."""
        assert compute_reward("1/2", r"\boxed{0.5}") == 1.0
        assert compute_reward("0.5", r"\boxed{1/2}") == 1.0
        assert compute_reward(r"\boxed{1/2}", r"\boxed{0.5}") == 1.0
        assert compute_reward(r"\boxed{0.5}", r"\boxed{1/2}") == 1.0
        assert compute_reward("50%", r"\boxed{1/2}") == 1.0

    def test_parentheses_negative(self):
        # [FUNCTION DOC]
        # "(100)" matches -100.
        """Test negative numbers in parentheses format."""
        assert compute_reward("(100)", r"\boxed{-100}") == 1.0


class TestHelperFunctions:
    # [CLASS DOC]
    # extract_boxed_answer and parse_number details.
    """Test helper functions used in reward computation."""

    def test_extract_boxed_answer(self):
        # [FUNCTION DOC]
        # Keeps the escaped percent inside the box; None without a box.
        """Test boxed answer extraction."""
        assert extract_boxed_answer(r"\boxed{6.280\%}") == r"6.280\%"
        assert extract_boxed_answer(r"\(\boxed{-0.606\%}\)") == r"-0.606\%"
        assert extract_boxed_answer("no box here") is None

    def test_parse_number_percentages(self):
        # [FUNCTION DOC]
        # Percentages parse to fractions, escaped or not.
        """Test percentage parsing."""
        assert abs(parse_number("6.28%") - 0.0628) < 1e-10
        assert abs(parse_number(r"6.280\%") - 0.0628) < 1e-10
        assert abs(parse_number("-0.606%") - (-0.00606)) < 1e-10

    def test_parse_number_ratios(self):
        # [FUNCTION DOC]
        # Decimals parse as floats.
        """Test ratio/decimal parsing."""
        assert parse_number("0.232") == 0.232
        assert parse_number("1.5") == 1.5

    def test_parse_number_fractions(self):
        # [FUNCTION DOC]
        # a/b parses to its quotient.
        """Test fraction parsing."""
        assert parse_number("1/2") == 0.5
        assert parse_number("3/4") == 0.75


class TestToleranceSettings:
    # [CLASS DOC]
    # tolerance and max_absolute_diff parameters.
    """Test the tolerance configuration."""

    def test_default_relative_tolerance(self):
        # [FUNCTION DOC]
        # 1% passes, 2% fails.
        """Default relative tolerance is 1% (0.01)."""
        # 100 vs 101 = 1% relative error, should match
        assert compute_reward("101", r"\boxed{100}") == 1.0
        # 100 vs 102 = 2% relative error, should fail
        assert compute_reward("102", r"\boxed{100}") == 0.0

    def test_custom_tolerance(self):
        # [FUNCTION DOC]
        # A larger tolerance still needs the absolute limit raised to pass.
        """Test with custom tolerance and absolute threshold parameters."""
        # With 2% tolerance but abs diff = 2 > 1.0, still fails
        assert compute_reward("102", r"\boxed{100}", tolerance=0.02) == 0.0
        # With 2% tolerance AND max_absolute_diff=3, passes
        assert (
            compute_reward("102", r"\boxed{100}", tolerance=0.02, max_absolute_diff=3.0)
            == 1.0
        )

    def test_absolute_tolerance_for_small_numbers(self):
        # [FUNCTION DOC]
        # Small numbers fail on relative error.
        """Small numbers must pass both relative (1%) AND absolute (1.0) checks."""
        # 0.5 vs 0.501 = 0.001 diff, 0.2% relative, passes both
        assert compute_reward("0.501", r"\boxed{0.5}") == 1.0
        # 0.5 vs 0.506 = 0.006 diff = 1.2% relative error, fails relative
        assert compute_reward("0.506", r"\boxed{0.5}") == 0.0

    def test_absolute_tolerance_for_large_numbers(self):
        # [FUNCTION DOC]
        # Differences within both limits pass; beyond them fail.
        """Large numbers must pass both relative (1%) AND absolute (1.0) checks."""
        # 100 vs 100.01 = 0.01 diff, 0.01% relative, passes both
        assert compute_reward("100.01", r"\boxed{100}") == 1.0
        # 100 vs 100.5 = 0.5 diff, 0.5% relative, passes both
        assert compute_reward("100.5", r"\boxed{100}") == 1.0
        # 100 vs 102 = 2 diff, 2% relative, fails both
        assert compute_reward("102", r"\boxed{100}") == 0.0


class TestBoundaryThresholds:
    # [CLASS DOC]
    # Values around 2.0.
    # KNOWN ISSUE: docstrings describe a 2.0 threshold switching between 0.001 and
    # 0.01 absolute tolerances, which the module-level description (1% relative
    # AND 1.0 absolute) does not mention; the asserted values pass either way.
    """Test boundary cases at the 2.0 threshold."""

    def test_at_threshold_exactly(self):
        # [FUNCTION DOC]
        # 2.0 and 2.001 match 2.0.
        """Test number exactly at 2.0 threshold."""
        assert compute_reward("2.0", r"\boxed{2.0}") == 1.0
        assert compute_reward("2.001", r"\boxed{2.0}") == 1.0

    def test_just_below_threshold(self):
        # [FUNCTION DOC]
        # 1.999 matches 2.0.
        """Test number just below 2.0 threshold (uses 0.001 tolerance)."""
        # 1.999 vs 2.0 = 0.001 diff, should match with 0.001 absolute tolerance
        assert compute_reward("1.999", r"\boxed{2.0}") == 1.0

    def test_just_above_threshold(self):
        # [FUNCTION DOC]
        # 2.001 matches 2.0.
        """Test number just above 2.0 threshold (uses 0.01 tolerance)."""
        # 2.001 vs 2.0 = 0.001 diff, should match with 0.01 absolute tolerance
        assert compute_reward("2.001", r"\boxed{2.0}") == 1.0


class TestScientificNotation:
    # [CLASS DOC]
    # Scientific notation parsing.
    """Test scientific notation handling."""

    def test_scientific_notation_basic(self):
        # [FUNCTION DOC]
        # e-notation matches decimal forms both ways.
        """Test basic scientific notation parsing."""
        assert compute_reward("1.23e-5", r"\boxed{0.0000123}") == 1.0
        assert compute_reward("1e6", r"\boxed{1000000}") == 1.0
        assert compute_reward("0.0000123", r"\boxed{1.23e-5}") == 1.0
        assert compute_reward(r"\boxed{1e6}", r"\boxed{1000000}") == 1.0

    def test_scientific_notation_percentages(self):
        # [FUNCTION DOC]
        # A percentage matches an e-notation fraction.
        """Test scientific notation with percentages."""
        # 1.23e-3% = 0.0000123
        assert compute_reward("0.00123%", r"\boxed{1.23e-5}") == 1.0


class TestExtremeValues:
    # [CLASS DOC]
    # Very large and very small values.
    """Test very large and very small numbers."""

    def test_very_large_numbers(self):
        # [FUNCTION DOC]
        # Large equal values match; 0.5% off fails on the absolute limit unless it is raised.
        """Test extremely large numbers with absolute threshold check."""
        assert compute_reward("1000000", r"\boxed{1000000}") == 1.0
        assert compute_reward("1000000", r"\boxed{1,000,000}") == 1.0
        assert compute_reward("1,000,000", r"\boxed{1000000}") == 1.0
        assert compute_reward("1e6", r"\boxed{1000000}") == 1.0
        # 1005000 vs 1000000 = 0.5% relative but abs diff = 5000 > 1.0, fails
        assert compute_reward("1005000", r"\boxed{1000000}") == 0.0
        # With custom max_absolute_diff, can pass
        assert (
            compute_reward("1005000", r"\boxed{1000000}", max_absolute_diff=10000)
            == 1.0
        )

    def test_very_small_decimals(self):
        # [FUNCTION DOC]
        # Tiny values match only within 1% relative.
        """Test very small decimal values."""
        assert compute_reward("0.00001", r"\boxed{0.00001}") == 1.0
        # 0.000011 vs 0.00001 = 10% relative error, fails
        assert compute_reward("0.000011", r"\boxed{0.00001}") == 0.0
        # Within 1% relative: 0.00001001 vs 0.00001 = 0.1% relative, passes
        assert compute_reward("0.00001001", r"\boxed{0.00001}") == 1.0

    def test_mixed_scale_comparison(self):
        # [FUNCTION DOC]
        # 1001 vs 1000 matches.
        """Test comparisons across different scales."""
        # 1,000 vs 1,001 = 0.1% relative error
        assert compute_reward("1001", r"\boxed{1000}") == 1.0


class TestWhitespaceAndFormatting:
    # [CLASS DOC]
    # Whitespace and LaTeX wrapper handling.
    """Test handling of whitespace and various formatting."""

    def test_extra_whitespace(self):
        # [FUNCTION DOC]
        # Surrounding whitespace is ignored.
        """Test answers with extra whitespace."""
        assert compute_reward("  6.28%  ", r"\boxed{6.280\%}") == 1.0
        assert compute_reward("100", r"\boxed{  100  }") == 1.0

    def test_multiple_latex_wrappers(self):
        # [FUNCTION DOC]
        # $$ and \[ \] wrappers are stripped.
        """Test various LaTeX wrapper formats."""
        # Double dollar signs
        assert compute_reward("25.14%", r"$$\boxed{25.144\%}$$") == 1.0
        # Display math mode
        assert compute_reward("0.232", r"\[\boxed{0.232}\]") == 1.0


class TestInvalidInputs:
    # [CLASS DOC]
    # Invalid inputs score 0 rather than raising.
    """Test handling of invalid or malformed inputs."""

    def test_empty_strings(self):
        # [FUNCTION DOC]
        # Empty prediction or ground truth scores 0.0.
        """Test empty string handling."""
        assert compute_reward("", r"\boxed{100}") == 0.0
        assert compute_reward("100", "") == 0.0

    def test_non_numeric_strings(self):
        # [FUNCTION DOC]
        # Non-numeric values score 0.0.
        """Test non-numeric string handling."""
        assert compute_reward("abc", r"\boxed{100}") == 0.0
        assert compute_reward("100", r"\boxed{abc}") == 0.0

    def test_malformed_fractions(self):
        # [FUNCTION DOC]
        # parse_number returns None for division by zero and a/b/c.
        """Test malformed fraction handling."""
        # Division by zero should not crash
        assert parse_number("1/0") is None
        # Invalid fraction format
        assert parse_number("1/2/3") is None

    def test_mixed_formats_mismatch(self):
        # [FUNCTION DOC]
        # 6.28 does not match 6.28%; 0.5 matches 1/2.
        """Test mismatched format types."""
        # Percentage vs plain number
        assert compute_reward("6.28", r"\boxed{6.28\%}") == 0.0
        # Fraction vs decimal (but these should match)
        assert compute_reward("0.5", r"\boxed{1/2}") == 1.0


class TestMultipleUnits:
    # [CLASS DOC]
    # Unit words and currency symbols.
    """Test various unit indicators."""

    def test_with_text_units(self):
        # [FUNCTION DOC]
        # Trailing unit words in ground truth are ignored.
        """Test numbers with text units like 'million'."""
        # The parse_number should extract just the number
        assert compute_reward("-77", r"\boxed{-77} million") == 1.0
        assert compute_reward("1.5", r"\boxed{1.5} billion") == 1.0

    def test_currency_symbols(self):
        # [FUNCTION DOC]
        # Dollar signs are ignored on either side.
        """Test with currency symbols."""
        assert compute_reward("$100", r"\boxed{100}") == 1.0
        assert compute_reward("100", r"\boxed{$100}") == 1.0
        assert compute_reward(r"\boxed{$100}", r"\boxed{100}") == 1.0
        assert compute_reward("$100.00", r"\boxed{100}") == 1.0


class TestPrecisionEdgeCases:
    # [CLASS DOC]
    # Leading zeros and percentages near 100%.
    """Test edge cases in precision matching."""

    def test_leading_zeros(self):
        # [FUNCTION DOC]
        # Leading/trailing zeros do not matter.
        """Test numbers with leading zeros."""
        assert compute_reward("0.50", r"\boxed{0.5}") == 1.0
        assert compute_reward("00.5", r"\boxed{0.5}") == 1.0

    def test_percentage_boundary(self):
        # [FUNCTION DOC]
        # Values within 0.1 points of 100% match.
        """Test percentage boundary cases near 100%."""
        assert compute_reward("100%", r"\boxed{100\%}") == 1.0
        assert compute_reward("99.9%", r"\boxed{100\%}") == 1.0
        assert compute_reward("100.1%", r"\boxed{100\%}") == 1.0


class TestPercentagePointsNotation:
    # [CLASS DOC]
    # Percentage-points fallback and multi-value boxes.
    """Test percentage points notation fallback (Bug fix #2)."""

    def test_percentage_points_basic(self):
        # [FUNCTION DOC]
        # "X%" matches bare X.
        """Test that '4.5%' matches '4.500' (both mean 4.5 percentage points)."""
        # Walmart tax rate differential example
        assert compute_reward("4.5%", r"\boxed{4.500}") == 1.0
        # GM tax rate difference example
        assert compute_reward("6.0%", r"\boxed{6.000}") == 1.0

    def test_percentage_points_general(self):
        # [FUNCTION DOC]
        # More percentage-point matches.
        """Test general percentage points matching."""
        assert compute_reward("6.28%", r"\boxed{6.28}") == 1.0
        assert compute_reward("10.5%", r"\boxed{10.5}") == 1.0

    def test_percentage_points_negative(self):
        # [FUNCTION DOC]
        # Negative percentage points match.
        """Test negative percentage points."""
        assert compute_reward("-2.5%", r"\boxed{-2.5}") == 1.0

    def test_multi_value_in_single_boxed(self):
        # [FUNCTION DOC]
        # A comma list matches a year-keyed box within tolerance.
        """Test comma-separated values inside single \\boxed{} with tolerance."""
        assert (
            compute_reward(
                "0.933, 0.931, 0.930",
                r"\boxed{2022:\ 0.933,\; 2023:\ 0.930,\; 2024:\ 0.931}",
            )
            == 1.0
        )


class TestMultiValueYearKeyMatching:
    # [CLASS DOC]
    # Year-keyed multi-value matching.
    """Test year-keyed order-independent matching and LaTeX whitespace handling."""

    def test_year_key_order_independence(self):
        # [FUNCTION DOC]
        # Keyed values match in any order; unkeyed values match positionally;
        # swapped values fail.
        """Year-labeled values match regardless of order; wrong values still fail."""
        gt = r"\boxed{2022: 0.100, 2023: 0.500, 2024: 0.900}"
        # Same order, reversed order, and unlabeled positional all work
        assert compute_reward("2022: 0.100, 2023: 0.500, 2024: 0.900", gt) == 1.0
        assert compute_reward("2024: 0.900, 2023: 0.500, 2022: 0.100", gt) == 1.0
        assert compute_reward("0.100, 0.500, 0.900", gt) == 1.0
        # Swapped values fail; wrong positional order fails
        assert compute_reward("2024: 0.100, 2023: 0.500, 2022: 0.900", gt) == 0.0
        assert compute_reward("0.900, 0.500, 0.100", gt) == 0.0
        # Negative values with reversed order
        gt_neg = r"\boxed{2024: -433275, 2023: -393364, 2022: -483361}"
        assert (
            compute_reward("2022: -483361, 2023: -393364, 2024: -433275", gt_neg) == 1.0
        )

    def test_year_range_keys_and_formats(self):
        # [FUNCTION DOC]
        # Year-range keys with "to", "->" or "-" match in any order.
        """Year-range keys (2022 to 2023) match with various arrow formats."""
        gt = r"\boxed{2022 to 2023: -0.002, 2023 to 2024: 0.046}"
        assert compute_reward("2022 to 2023: -0.002, 2023 to 2024: 0.046", gt) == 1.0
        assert compute_reward("2023 to 2024: 0.046, 2022 to 2023: -0.002", gt) == 1.0
        assert compute_reward("2022 -> 2023: -0.002, 2023 -> 2024: 0.046", gt) == 1.0
        assert compute_reward("2022-2023: -0.002, 2023-2024: 0.046", gt) == 1.0

    def test_latex_whitespace_in_multi_value(self):
        # [FUNCTION DOC]
        # LaTeX spacing commands inside boxed lists are ignored.
        r"""LaTeX whitespace (\  and \;) in multi-value answers parses correctly."""
        assert (
            compute_reward("1.107, 1.031, 0.926", r"\boxed{1.107,\ 1.031,\ 0.926}")
            == 1.0
        )
        assert compute_reward("8908, 7960, 6209", r"\boxed{8908,\ 7960,\ 6209}") == 1.0
        assert (
            compute_reward(
                "2022: 0.933, 2023: 0.930, 2024: 0.931",
                r"\boxed{2022:\ 0.933,\; 2023:\ 0.930,\; 2024:\ 0.931}",
            )
            == 1.0
        )


# ---------------------------------------------------------------------------
# Integration tests (require downloaded data)
# ---------------------------------------------------------------------------

# Data path relative to repo root
DATA_PATH = str(Path(__file__).parent.parent.parent / "envs" / "finqa_env" / "data")

_data_available = os.path.isfile(
    os.path.join(DATA_PATH, "benchmark_questions", "finqa.csv")
)

try:
    import pandas  # noqa: F401

    _pandas_available = True
except ImportError:
    _pandas_available = False

_integration_skip = not (_data_available and _pandas_available)
_integration_reason = "requires downloaded data and pandas"


@pytest.mark.skipif(_integration_skip, reason=_integration_reason)
class TestTools:
    # [CLASS DOC]
    # FinQATools tests on downloaded data (skipped without data or pandas).
    """Test tool implementations."""

    @pytest.fixture
    def tools(self):
        # [FUNCTION DOC]
        # Fixture: FinQATools over the repo data directory.
        from envs.finqa_env.server.tools import FinQATools

        return FinQATools(DATA_PATH)

    def test_get_available_companies(self, tools):
        # [FUNCTION DOC]
        # Company list is non-empty and includes alphabet.
        companies = tools.get_available_companies()
        assert len(companies) > 0
        assert "alphabet" in companies

    def test_get_descriptions(self, tools):
        # [FUNCTION DOC]
        # Descriptions list GAAP tables without error.
        result = tools.get_descriptions("alphabet")
        assert "Error" not in result
        assert "us_gaap_" in result  # Should contain GAAP table names

    def test_get_descriptions_invalid_company(self, tools):
        # [FUNCTION DOC]
        # Unknown company returns an error string.
        result = tools.get_descriptions("nonexistent_company")
        assert "Error" in result

    def test_get_table_info(self, tools):
        # [FUNCTION DOC]
        # Table info includes column_dtypes.
        result = tools.get_table_info(
            "alphabet",
            "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock",
        )
        assert "Error" not in result
        assert "column_dtypes" in result

    def test_sql_query_no_filter(self, tools):
        # [FUNCTION DOC]
        # A query without a filter returns an error string.
        result = tools.sql_query("alphabet", "some_table", "SELECT * FROM some_table")
        assert "Error" in result


class TestToolsPathTraversal:
    # [CLASS DOC]
    # Path traversal protection in FinQATools with synthetic data in tmp_path.
    """Agent-supplied company/table names must not escape the data directory (CWE-22).

    Self-contained (synthetic data, no downloaded dataset needed) so it runs in CI.
    """

    @pytest.fixture
    def tools(self, tmp_path):
        # [FUNCTION DOC]
        # Fixture: creates input_companies/acme/revenue.json and an out-of-tree
        # secret.json in tmp_path; returns FinQATools over tmp_path. Skips without pandas.
        pytest.importorskip("pandas")
        from envs.finqa_env.server.tools import FinQATools

        company = tmp_path / "input_companies" / "acme"
        company.mkdir(parents=True)
        (company / "revenue.json").write_text('[{"x": 1}]')
        # A file the agent must never be able to reach, placed outside input_companies/.
        (tmp_path / "secret.json").write_text('[{"password": "leaked-secret"}]')
        return FinQATools(str(tmp_path))

    def test_legit_access_still_works(self, tools):
        # [FUNCTION DOC]
        # Normal company access works without leaking the secret.
        assert "Error" not in tools.get_descriptions("acme")
        assert "leaked" not in tools.sql_query(
            "acme", "revenue", "SELECT x FROM revenue WHERE x = 1"
        )

    def test_get_descriptions_rejects_traversal(self, tools):
        # [FUNCTION DOC]
        # Traversing company names return errors.
        for evil in ["..", "../..", "../../../../etc"]:
            result = tools.get_descriptions(evil)
            assert "Error" in result
            assert "secret" not in result

    def test_sql_query_rejects_company_traversal(self, tools):
        # [FUNCTION DOC]
        # Company traversal cannot read the secret.
        result = tools.sql_query(
            "..", "secret", "SELECT password FROM secret WHERE 1 = 1"
        )
        assert "leaked-secret" not in result

    def test_sql_query_rejects_table_traversal(self, tools):
        # [FUNCTION DOC]
        # Table traversal cannot read the secret.
        result = tools.sql_query(
            "acme", "../../secret", "SELECT password FROM x WHERE 1 = 1"
        )
        assert "leaked-secret" not in result

    def test_get_table_info_rejects_traversal(self, tools):
        # [FUNCTION DOC]
        # Traversal in get_table_info returns an error.
        assert "Error" in tools.get_table_info("../..", "whatever")


class TestResetSeed:
    # [CLASS DOC]
    # reset(seed=...) reproducibility with a synthetic 20-question CSV.
    """reset(seed=...) must make the question order reproducible.

    Self-contained (synthetic data, no downloaded dataset needed) so it runs in CI.
    """

    @pytest.fixture
    def data_path(self, tmp_path):
        # [FUNCTION DOC]
        # Fixture: writes benchmark_questions/finqa.csv with 20 rows into tmp_path;
        # returns the directory. Skips without pandas.
        pytest.importorskip("pandas")
        questions = tmp_path / "benchmark_questions"
        questions.mkdir()
        rows = ["id,user_query,company,question,answer"]
        rows += [f"{i},query {i},acme,question {i},\\boxed{{{i}}}" for i in range(20)]
        (questions / "finqa.csv").write_text("\n".join(rows) + "\n")
        return str(tmp_path)

    @staticmethod
    def _make_env(data_path):
        # [FUNCTION DOC]
        # HELPER: returns a FinQAEnvironment over the data path.
        from envs.finqa_env.server.finqa_environment import FinQAEnvironment

        return FinQAEnvironment(data_path=data_path)

    @staticmethod
    def _questions_after_seed(env, seed, n=5):
        # [FUNCTION DOC]
        # HELPER: returns questions from one seeded reset followed by n-1 unseeded resets.
        """One seeded reset followed by n - 1 unseeded ones."""
        questions = [env.reset(seed=seed).metadata["question"]]
        questions += [env.reset().metadata["question"] for _ in range(n - 1)]
        return questions

    def test_same_seed_same_question_across_instances(self, data_path):
        # [FUNCTION DOC]
        # Same seed gives the same first question on fresh instances.
        questions = {
            self._make_env(data_path).reset(seed=1234).metadata["question"]
            for _ in range(5)
        }
        assert len(questions) == 1

    def test_same_seed_same_sequence_across_instances(self, data_path):
        # [FUNCTION DOC]
        # Same seed gives the same question sequence across instances.
        first = self._questions_after_seed(self._make_env(data_path), seed=1234)
        second = self._questions_after_seed(self._make_env(data_path), seed=1234)
        assert first == second

    def test_reseeding_restarts_the_sequence(self, data_path):
        # [FUNCTION DOC]
        # Reseeding one instance restarts the same sequence.
        env = self._make_env(data_path)
        first = self._questions_after_seed(env, seed=1234)
        second = self._questions_after_seed(env, seed=1234)
        assert first == second

    def test_different_seeds_give_different_sequences(self, data_path):
        # [FUNCTION DOC]
        # Different seeds give different sequences.
        env = self._make_env(data_path)
        assert self._questions_after_seed(env, seed=1) != self._questions_after_seed(
            env, seed=2
        )


@pytest.mark.skipif(_integration_skip, reason=_integration_reason)
class TestEnvironment:
    # [CLASS DOC]
    # FinQAEnvironment tests on downloaded data (skipped without data or pandas).
    """Test environment logic using MCP actions."""

    @pytest.fixture
    def env(self):
        # [FUNCTION DOC]
        # Fixture: FinQAEnvironment over the repo data with max_steps 10.
        from envs.finqa_env.server.finqa_environment import FinQAEnvironment

        return FinQAEnvironment(data_path=DATA_PATH, max_steps=10)

    def test_reset(self, env):
        # [FUNCTION DOC]
        # reset metadata has question, company, step_count 0 and not done.
        from openenv.core.env_server.types import Observation

        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.metadata["question"] != ""
        assert obs.metadata["company"] != ""
        assert obs.metadata["step_count"] == 0
        assert obs.done is False

    def test_list_tools(self, env):
        # [FUNCTION DOC]
        # ListToolsAction returns tools without ending the episode.
        from openenv.core.env_server.mcp_types import ListToolsAction

        env.reset()
        obs = env.step(ListToolsAction())
        assert obs.done is False
        assert "tools" in obs.metadata or hasattr(obs, "tools")

    def test_step_get_descriptions(self, env):
        # [FUNCTION DOC]
        # get_descriptions call does not end the episode.
        from openenv.core.env_server.mcp_types import CallToolAction

        obs = env.reset()
        company = obs.metadata["company"]
        action = CallToolAction(
            tool_name="get_descriptions", arguments={"company_name": company}
        )
        obs = env.step(action)
        assert obs.done is False

    def test_step_submit_answer(self, env):
        # [FUNCTION DOC]
        # submit_answer ends the episode with reward 0.0 or 1.0.
        from openenv.core.env_server.mcp_types import CallToolAction

        env.reset()
        action = CallToolAction(
            tool_name="submit_answer", arguments={"answer": "6.118"}
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward is not None
        assert obs.reward in [0.0, 1.0]

    def test_max_steps_termination(self, env):
        # [FUNCTION DOC]
        # Hitting max steps ends the episode with reward 0.0.
        from openenv.core.env_server.mcp_types import CallToolAction

        env.reset()
        for _ in range(10):
            action = CallToolAction(
                tool_name="get_descriptions", arguments={"company_name": "test"}
            )
            obs = env.step(action)
            if obs.done:
                break

        assert obs.done is True
        assert obs.reward == 0.0  # No answer submitted

    def test_state_property(self, env):
        # [FUNCTION DOC]
        # State is a FinQAState with episode id and current question.
        from envs.finqa_env.models import FinQAState

        env.reset()
        state = env.state
        assert isinstance(state, FinQAState)
        assert state.episode_id is not None
        assert state.current_question != ""

    def test_repeated_resets(self, env):
        # [FUNCTION DOC]
        # Repeated resets each produce a fresh question.
        """Test that multiple resets produce valid state each time."""
        for _ in range(3):
            obs = env.reset()
            assert obs.done is False
            assert obs.metadata["question"] != ""
            assert obs.metadata["company"] != ""

    def test_invalid_tool_name(self, env):
        # [FUNCTION DOC]
        # Unknown tools do not crash and either keep the episode going or report an error.
        """Test calling a tool that doesn't exist."""
        from openenv.core.env_server.mcp_types import CallToolAction

        env.reset()
        action = CallToolAction(tool_name="nonexistent_tool", arguments={})
        obs = env.step(action)
        # Should not crash; returns error in metadata
        assert obs.done is False or "error" in str(obs.metadata).lower()

    def test_empty_tool_args(self, env):
        # [FUNCTION DOC]
        # Missing arguments do not crash.
        """Test calling a tool with missing required arguments."""
        from openenv.core.env_server.mcp_types import CallToolAction

        env.reset()
        action = CallToolAction(tool_name="get_descriptions", arguments={})
        obs = env.step(action)
        # Should not crash
        assert isinstance(obs.done, bool)

    def test_state_consistency_after_steps(self, env):
        # [FUNCTION DOC]
        # Episode id stays fixed and step_count increments.
        """Test that state is consistent after multiple steps."""
        from openenv.core.env_server.mcp_types import CallToolAction

        env.reset()
        initial_episode_id = env.state.episode_id

        action = CallToolAction(
            tool_name="get_descriptions", arguments={"company_name": "alphabet"}
        )
        env.step(action)
        assert env.state.episode_id == initial_episode_id
        assert env.state.step_count == 1

        env.step(action)
        assert env.state.step_count == 2

    def test_sql_injection_attempt(self, env):
        # [FUNCTION DOC]
        # An injection-style query does not crash.
        # KNOWN ISSUE: the docstring says an error should be returned but only the
        # type of done is asserted.
        """Test that SQL injection attempts are handled safely."""
        from openenv.core.env_server.mcp_types import CallToolAction

        env.reset()
        action = CallToolAction(
            tool_name="sql_query",
            arguments={
                "company_name": "alphabet",
                "table_name": "test; DROP TABLE users;--",
                "query": "SELECT * FROM test WHERE 1=1; DROP TABLE users;--",
            },
        )
        obs = env.step(action)
        # Should not crash, should return an error
        assert isinstance(obs.done, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
