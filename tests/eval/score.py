#!/usr/bin/env python3
"""Pure answer extractors and per-task scorers for the M5 harness.

The extractors are heuristics tuned for short, greedy generations with a final-answer convention:
math items are asked for a final integer; multiple-choice items for a single letter; generation items are
checked by required substrings.
"""
import re


def extract_last_integer(text):
    """Return the final integer as a normalized string, or None.

    Preference order: \\boxed{N}, an explicit 'answer is N' / 'answer: N', **N**, else the last integer.
    """
    for pat in (r"\\boxed\{\s*(-?\d+)\s*\}",
                r"(?i)answer\s*(?:is|:)?\s*\**\s*(-?\d+)",
                r"\*\*\s*(-?\d+)\s*\*\*"):
        m = re.findall(pat, text)
        if m:
            return str(int(m[-1]))
    m = re.findall(r"-?\d+", text)
    return str(int(m[-1])) if m else None


def extract_mc_letter(text, options="ABCD"):
    """Return the chosen option letter (uppercased) or None."""
    m = re.findall(rf"(?i)answer\s*(?:is|:)?\s*\**\s*\(?([{options}])\b", text)
    if m:
        return m[-1].upper()
    m = re.findall(rf"(?m)^\s*\(?([{options}])\s*[\).:]", text)
    if m:
        return m[-1].upper()
    m = re.findall(rf"\b([{options}])\b", text)
    return m[0].upper() if m else None


def score_math(text, gold):
    return extract_last_integer(text) == str(int(gold))


def score_mc(text, gold):
    return extract_mc_letter(text) == str(gold).upper()


def score_gen(text, checks):
    low = text.lower()
    return all(str(c).lower() in low for c in checks)
