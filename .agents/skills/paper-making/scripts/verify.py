#!/usr/bin/env python3
"""Deterministic verification tool for academic & scientific papers.

Checks LaTeX balance, academic style (weasel words), missing placeholders,
and calculates exact metric deltas.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


def verify_latex_balance(text: str) -> List[str]:
    """Check balancing of LaTeX math delimiters ($, $$, \\begin...\\end) and braces."""
    errors = []

    # Check $$ block delimiters
    block_delimiters = re.findall(r"\$\$", text)
    if len(block_delimiters) % 2 != 0:
        errors.append("[LATEX_ERROR] Unmatched '$$' block math delimiters (odd count).")

    # Remove blocks before counting inline $
    text_no_blocks = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    inline_delimiters = re.findall(r"(?<!\\)\$", text_no_blocks)
    if len(inline_delimiters) % 2 != 0:
        errors.append("[LATEX_ERROR] Unmatched inline '$' math delimiters (odd count).")

    # Check braces {} in math expressions
    math_segments = re.findall(r"\$\$.*?\$\$|\$.*?\$", text, flags=re.DOTALL)
    for seg in math_segments:
        open_braces = len(re.findall(r"(?<!\\)\{", seg))
        close_braces = len(re.findall(r"(?<!\\)\}", seg))
        if open_braces != close_braces:
            snippet = seg[:45].replace("\n", " ")
            errors.append(f"[LATEX_ERROR] Unbalanced braces '{{}}' in math segment: {snippet}...")

    # Check \begin{env} and \end{env}
    begins = re.findall(r"\\begin\{([a-zA-Z*]+)\}", text)
    ends = re.findall(r"\\end\{([a-zA-Z*]+)\}", text)
    if len(begins) != len(ends):
        errors.append(f"[LATEX_ERROR] Environment count mismatch: {len(begins)} \\begin vs {len(ends)} \\end.")

    return errors


def lint_academic_style(text: str) -> List[str]:
    """Check for informal tone, weasel words, and unhedged claims."""
    warnings = []

    informal_patterns: List[Tuple[str, str]] = [
        (r"\b(very|extremely|super|hugely|tremendously)\b", "Exaggerated adverb; replace with exact quantitative metrics."),
        (r"\b(obviously|clearly|of course|undoubtedly)\b", "Subjective assertion; back with empirical evidence or citations."),
        (r"\bprove[sd]?\b", "In empirical AI/CS, use 'demonstrate', 'show', or 'empirically validate' instead of 'prove'."),
        (r"\b(cực kỳ|rất tốt|hoàn hảo|tuyệt vời|chắc chắn là|hiển nhiên)\b", "Informal/hyperbolic Vietnamese phrasing unsuitable for scientific papers."),
    ]

    for pattern, advice in informal_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            unique_words = sorted(list(set(m.lower() for m in matches)))
            warnings.append(f"[STYLE_WARNING] Found informal/weasel terms {unique_words}: {advice}")

    return warnings


def check_placeholders_and_citations(text: str) -> List[str]:
    """Check for unresolved placeholders and empty citation brackets."""
    issues = []

    # Placeholders
    placeholders = re.findall(r"(\[TODO.*?\]|\[MISSING_DATA.*?\]|\[\?\?\?\]|\bTBD\b)", text)
    if placeholders:
        issues.append(f"[DATA_WARNING] Found unresolved placeholders: {placeholders}")

    # Empty citations
    empty_cites = re.findall(r"(\\cite\{\s*\}|\[\s*\])", text)
    if empty_cites:
        issues.append("[CITATION_ERROR] Found empty citation brackets (\\cite{} or []).")

    return issues


def calculate_metric_delta(baseline: float, proposed: float, metric_name: str = "Accuracy") -> str:
    """Calculate exact absolute and relative percentage improvements."""
    abs_diff = proposed - baseline
    rel_diff = (abs_diff / baseline * 100.0) if baseline != 0 else 0.0
    sign = "+" if abs_diff >= 0 else ""
    return (
        f"Metric: {metric_name} | Baseline: {baseline:.2f} | Proposed: {proposed:.2f} | "
        f"Absolute Delta: {sign}{abs_diff:.2f} | Relative Change: {sign}{rel_diff:.2f}%"
    )


def audit_text(text: str) -> Tuple[bool, List[str]]:
    """Run full audit suite on manuscript text."""
    reports = []
    latex_errors = verify_latex_balance(text)
    style_warnings = lint_academic_style(text)
    data_issues = check_placeholders_and_citations(text)

    reports.extend(latex_errors)
    reports.extend(data_issues)
    reports.extend(style_warnings)

    has_errors = any("[LATEX_ERROR]" in r or "[CITATION_ERROR]" in r for r in reports)
    return (not has_errors), reports


def main():
    parser = argparse.ArgumentParser(description="Scientific Manuscript Deterministic Verifier")
    parser.add_argument("file", nargs="?", help="Path to markdown or latex manuscript file")
    parser.add_argument("--text", help="Raw manuscript text snippet")
    parser.add_argument("--calc-delta", nargs=3, metavar=("BASELINE", "PROPOSED", "METRIC"),
                        help="Calculate delta between baseline and proposed values")

    args = parser.parse_args()

    if args.calc_delta:
        try:
            b = float(args.calc_delta[0])
            p = float(args.calc_delta[1])
            m = args.calc_delta[2]
            print(calculate_metric_delta(b, p, m))
            sys.exit(0)
        except ValueError:
            print("Error: Baseline and Proposed values must be numbers.")
            sys.exit(1)

    content = ""
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        content = path.read_text(encoding="utf-8")
    elif args.text:
        content = args.text
    else:
        parser.print_help()
        sys.exit(1)

    passed, reports = audit_text(content)
    if not reports:
        print("PASS: Manuscript passed all deterministic checks (LaTeX syntax, tone, and placeholders).")
        sys.exit(0)
    else:
        status_line = "FAIL (Critical Errors)" if not passed else "WARNING (Actionable Improvements)"
        print(f"{status_line}:")
        for item in reports:
            print(f"  - {item}")
        sys.exit(1 if not passed else 0)


if __name__ == "__main__":
    main()
