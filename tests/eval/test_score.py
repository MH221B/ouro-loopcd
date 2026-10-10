#!/usr/bin/env python3
"""Self-test for score.py (plain asserts; exit non-zero on failure)."""
import sys
from score import extract_last_integer, extract_mc_letter, score_math, score_mc, score_gen

def eq(name, got, want):
    ok = got == want
    print(("ok   " if ok else "FAIL ") + f"{name}: got={got!r} want={want!r}")
    return ok

def main():
    fails = 0
    fails += not eq("fenced answer", extract_last_integer("... compute ... so the answer is **42**"), "42")
    fails += not eq("boxed", extract_last_integer("Therefore \\boxed{24}."), "24")
    fails += not eq("explicit", extract_last_integer("The final answer is 391."), "391")
    fails += not eq("negative", extract_last_integer("we get -3 in the end"), "-3")
    fails += not eq("last int fallback", extract_last_integer("first 2 then 55"), "55")
    fails += not eq("no integer", extract_last_integer("no number here"), None)

    fails += not eq("answer letter", extract_mc_letter("The answer is C."), "C")
    fails += not eq("option letter", extract_mc_letter("B) Mercury"), "B")
    fails += not eq("no letter", extract_mc_letter("none of these"), None)

    fails += not eq("score_math pass", score_math("...answer is 42", 42), True)
    fails += not eq("score_math fail", score_math("...answer is 41", 42), False)
    fails += not eq("score_mc pass", score_mc("Answer: B", "B"), True)
    fails += not eq("score_mc fail", score_mc("Answer: A", "B"), False)
    fails += not eq("score_gen pass", score_gen("def fib(n):\n    return n", ["def fib", "return"]), True)
    fails += not eq("score_gen fail", score_gen("def f(): pass", ["def fib"]), False)

    print("SCORE: FAIL" if fails else "SCORE: PASS")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
