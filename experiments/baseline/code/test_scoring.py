"""Regression tests for scoring.py.

Every case here is either (a) a counterexample the 2026-09-11 review produced
against the v1 normaliser, or (b) a case where the v1 rule merged two values
that are not equal.  A scoring change that breaks one of these must not ship.

Run:  python3 -m pytest tests/test_scoring.py -q
      python3 tests/test_scoring.py          (no pytest needed)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import (norm_answer, norm_key, gold_key, correct,  # noqa: E402
                     parse_schema, answer_class, semantic_class)

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------- review 5.2
@case("mbpp_106: parenthesised string gold equals bare string reply")
def _():
    g = "('Not matched!')"
    assert gold_key(g, "mbpp") == norm_answer("Not matched!", "mbpp")
    assert correct("Not matched!", g, "mbpp")
    # A quoted reply is the same Python string too.
    assert correct("'Not matched!'", g, "mbpp")
    assert correct("('Not matched!')", g, "mbpp")


@case("mbpp_129: parenthesised string gold equals bare string reply")
def _():
    g = "('pythonprogram123')"
    assert correct("pythonprogram123", g, "mbpp")
    assert correct("pythonprogram123\n", g, "mbpp")


@case("mbpp_131: tuple spacing does not matter")
def _():
    g = "(0,6)"
    assert correct("(0, 6)", g, "mbpp")
    assert correct("(0,6)", g, "mbpp")
    assert not correct("(0, 7)", g, "mbpp")


@case("review 10.2: v1 merged '1 m' and '1 cm'")
def _():
    g = "1 m"
    assert not correct("1 cm", g, "mbpp")
    # ... but the number itself is still the same value where the benchmark is
    # numeric, which is the GSM8K rule, not the MBPP rule.
    assert norm_answer("1 cm", "mbpp") != norm_answer("1 m", "mbpp")


@case("review 10.2: v1 merged two different KEYs via the answer rule")
def _():
    a = "a. add the amounts"
    b = "a. multiply the amounts"
    assert norm_key(a) != norm_key(b)


# ---------------------------------------------------------------- correctness
@case("mbpp: type is part of the value")
def _():
    assert not correct("1", "(1,)", "mbpp")          # int vs tuple
    assert not correct("(1, 2)", "[1, 2]", "mbpp")   # tuple vs list
    assert correct("[1, 2]", "[1, 2]", "mbpp")
    assert correct("True", "True", "mbpp")
    assert not correct("true", "True", "mbpp")       # not a Python literal
    # A nested string keeps its identity: ('a',) is not the string a, and the
    # tuple type is not erased by the string unification.
    assert not correct("a", "('a',)", "mbpp")
    assert not correct("('a',)", "'a'", "mbpp")
    assert correct("('a',)", "('a',)", "mbpp")


@case("mbpp: bare vs quoted scalar text is cosmetic (documented assumption)")
def _():
    # 35/50 MBPP golds are plain values and several are quoted strings such as
    # 'Not matched!'.  A reply that omits the quotes is the same value; this is
    # the exact v1 defect.  Every numeric MBPP gold is unquoted, so the
    # converse case cannot arise in this task set.
    assert correct("'1'", "1", "mbpp")
    assert correct("Not matched!", "'Not matched!'", "mbpp")
    assert correct("'Not matched!'", "('Not matched!')", "mbpp")


@case("mbpp: numeric equalities that must hold")
def _():
    assert correct("6", "6", "mbpp")
    assert correct("6.0", "6", "mbpp")
    assert correct("0.50", "0.5", "mbpp")
    assert not correct("6", "7", "mbpp")


@case("gsm8k: unit and percent stripping, thousands separators")
def _():
    assert correct("160 minutes", "160", "gsm8k")
    assert correct("160%", "160", "gsm8k")
    assert correct("1,200", "1200", "gsm8k")
    assert correct("$6", "6", "gsm8k")
    assert correct(" 168 ", "168", "gsm8k")
    assert correct("168.0", "168", "gsm8k")
    assert not correct("168", "169", "gsm8k")


@case("gsm8k: a reply with no number is not silently coerced")
def _():
    # v1 lowercased and stripped to produce a text class; that is fine, but it
    # must not compare equal to a numeric gold.
    assert not correct("about a hundred and sixty", "160", "gsm8k")


@case("mmlu: option letters in every observed surface form")
def _():
    for text in ["B", "B.", "B)", "**B**", "b", " B ", "B. False, False",
                 "B) True", "answer: B", "FINAL ANSWER: B"]:
        assert correct(text, "B", "mmlu"), text
    assert not correct("C", "B", "mmlu")
    assert norm_answer("B. False, False", "mmlu") == ("option", "B")


@case("mmlu: an option rule must not eat a two-word prose answer")
def _():
    # 'a' alone is option A; 'a cat' is not an option letter.
    assert norm_answer("A", "mmlu") == ("option", "A")
    assert norm_answer("a cat", "mmlu")[0] == "opaque"


@case("answer extraction: last schema block, not last field of each kind")
def _():
    text = "ANSWER: 5\nKEY: first step\nsome more reasoning\nANSWER: 7\nKEY: second step"
    a, k = parse_schema(text)
    assert (a, k) == ("7", "second step"), (a, k)


@case("answer extraction: none present")
def _():
    assert parse_schema("no schema here") == (None, None)
    assert parse_schema("") == (None, None)
    assert parse_schema(None) == (None, None)


@case("verdict vs answer+KEY are different granularities")
def _():
    assert answer_class("7", "gsm8k") == answer_class("7", "gsm8k")
    assert semantic_class("7", "add the parts", "gsm8k") != \
        semantic_class("7", "multiply the parts", "gsm8k")
    # verdict collapses them
    assert answer_class("7", "gsm8k") == answer_class("7", "gsm8k")


@case("missing fields produce None, never a shared class")
def _():
    assert answer_class(None, "gsm8k") is None
    assert norm_key(None) is None
    assert semantic_class(None, "x", "gsm8k") is None
    assert semantic_class("7", None, "gsm8k") == (("number", "7"), None)


@case("markdown and schema residue are removed but punctuation is not")
def _():
    assert norm_answer("**ANSWER: 6**", "gsm8k") == ("number", "6")
    assert norm_answer("`42`", "gsm8k") == ("number", "42")
    # A trailing period on a tuple must not change the value.
    assert correct("(0, 6).", "(0,6)", "mbpp")
    # ... but a comma inside a numeric string on MBPP is a tuple separator.
    assert norm_answer("(0,6)", "mbpp") != norm_answer("(06)", "mbpp")


def main():
    failed = []
    for name, fn in CASES:
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, exc))
            print(f"FAIL  {name}")
            print(f"      {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(CASES) - len(failed)}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
