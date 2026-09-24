"""Benchmark-specific answer scoring and separate answer/KEY normalisation.

Repairs the two defects confirmed in the 2026-09-11 review of the v1 code:

  1. v1 used one normaliser for every benchmark.  It stripped commas and
     lowercased unconditionally, so the MBPP gold ``('Not matched!')`` and the
     correct reply ``Not matched!`` were scored as different.  At least 269
     correct CoT replies were counted wrong.  Scoring is now dispatched per
     benchmark: numeric for GSM8K, option-letter for MMLU, and *typed
     recursive literal comparison* for MBPP.

  2. v1 normalised the KEY with the rule built for the ANSWER, which mapped
     both ``a. add the amounts`` and ``a. multiply the amounts`` to ``a``.
     ``norm_answer`` and ``norm_key`` are now separate, and the multi-choice
     rule is confined to answers.

Every rule here has a regression test in ``tests/test_scoring.py`` built from
the review's own counterexamples.  Nothing in this module is allowed to merge
two values that are not equal under the benchmark's own semantics.
"""
from __future__ import annotations

import ast
import decimal
import re

__all__ = [
    "norm_answer", "norm_key", "gold_key", "correct", "parse_schema",
    "answer_class", "semantic_class", "MMLU_RE",
]

MMLU_RE = re.compile(r"^\s*\**\s*([A-Da-d])\s*\**\s*(?:[.):]\s*.*)?$")
_NUM_RE = re.compile(
    r"^([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:%|[A-Za-z][A-Za-z .\-/]*)?$"
)
_MD_HEAD = re.compile(r"^(?:\*\*|`|#|>)+\s*")
_MD_TAIL = re.compile(r"(?:\*\*|`)+$")
_ANSWER_PREFIX = re.compile(r"^(?:final\s+)?answer\s*[:：]\s*", re.I)
_KEY_PREFIX = re.compile(r"^(?:key|step|reason(?:ing)?)\s*[:：]\s*", re.I)


# --------------------------------------------------------------------------
# generic cleanup
# --------------------------------------------------------------------------
def _strip_schema(text):
    """Remove markdown emphasis and a leading ``ANSWER:``/``KEY:`` residue.

    Markdown markers are stripped *before* the field prefix, otherwise
    ``**ANSWER: 6**`` leaves ``answer: 6`` as the value.
    """
    if text is None:
        return None
    s = str(text).strip()
    s = _MD_HEAD.sub("", s)
    s = _MD_TAIL.sub("", s).strip()
    s = _ANSWER_PREFIX.sub("", s)
    s = _KEY_PREFIX.sub("", s)
    s = _MD_HEAD.sub("", s)
    s = _MD_TAIL.sub("", s)
    return s.strip() or None


def _decimal(text):
    try:
        d = decimal.Decimal(text)
    except Exception:
        return None
    return d if d.is_finite() else None


def _number_key(d):
    """Canonical string for a finite decimal, so 160 == 160.0 == 160.00."""
    if d == 0:
        d = abs(d)
    return ("number", format(d.normalize(), "f"))


# --------------------------------------------------------------------------
# ANSWER
# --------------------------------------------------------------------------
def norm_answer(text, bench):
    """Map an answer string to a comparable hashable key, per benchmark.

    Returns ``None`` when no answer can be extracted; callers must treat that
    as a missing observation rather than as its own class.
    """
    s = _strip_schema(text)
    if s is None:
        return None
    s = s.strip().strip("\"'").strip()

    if bench == "mmlu":
        m = MMLU_RE.match(s)
        if m:
            return ("option", m.group(1).upper())
        # An MMLU reply that is not an option letter is opaque, not an error:
        # keep the cleaned text so it forms its own class.
        return ("opaque", " ".join(s.split()).casefold())

    if bench == "gsm8k":
        t = s.replace("$", "").replace(",", "").replace("%", "").strip()
        d = _decimal(t)
        if d is not None:
            return _number_key(d)
        m = _NUM_RE.match(s.replace("$", ""))
        if m:
            d = _decimal(m.group(1))
            if d is not None:
                return _number_key(d)
        return ("text", " ".join(s.split()).casefold())

    if bench == "mbpp":
        # Surrounding single/double quotes around the *whole* value are a
        # representation choice, so try the quoted form first (it lets a
        # literal like `'a, b, c'` stay one string) and only then the bare
        # form.  Trailing punctuation is preserved because it is part of the
        # value for MBPP, not sentence punctuation.
        return _literal_key(s)

    raise ValueError(f"unknown benchmark {bench!r}")


def _literal_key(s):
    """Typed recursive comparison for MBPP-style returned values.

    ``('Not matched!')`` and ``Not matched!`` are the same Python value: a
    one-element parenthesised string is *not* a tuple.  ``(0,6)`` and
    ``(0, 6)`` are the same tuple.  Both are handled by parsing the literal
    and comparing the resulting objects, instead of deleting characters.

    When the text is not a Python literal we cannot tell whether the model
    emitted a *string value* without quotes or simply wrote prose.  Returning
    ``("raw", text)`` would make ``Not matched!`` unequal to the gold
    ``('Not matched!')`` and reintroduce the very bug this module repairs, so
    an unparseable text is carried as ``("text", ...)`` and unified with the
    corresponding string value in :func:`_unify_strings`.  That unification is
    applied *only* to single scalar strings, never to nested structures, so it
    cannot merge two distinct tuples or a tuple with a bare value.
    """
    cleaned = s.strip()
    # A code block may wrap the value.
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*|\s*```$", "", cleaned).strip()
    if not cleaned:
        return None

    obj = _try_literal(cleaned)
    if obj is _UNPARSED:
        # A sentence-ending period may follow the value; retry without it
        # before falling back, so `(0, 6).` still parses as a tuple.
        trimmed = cleaned.rstrip(".").strip()
        if trimmed and trimmed != cleaned:
            obj = _try_literal(trimmed)
            if obj is not _UNPARSED:
                return _unify_strings(_value_key(obj))
        return ("text", " ".join(cleaned.split()))
    return _unify_strings(_value_key(obj))


_UNPARSED = object()


def _unify_strings(key):
    """Collapse ``('string', s)`` to ``('text', s)`` so quoting is cosmetic.

    Only a top-level string is touched.  ``('tuple', (('string','a'),))`` is
    returned unchanged, which is what keeps ``('a',)`` distinct from ``a``.
    """
    if isinstance(key, tuple) and len(key) == 2 and key[0] == "string":
        return ("text", key[1])
    return key


def _try_literal(text):
    try:
        return ast.literal_eval(text)
    except Exception:
        return _UNPARSED


def _value_key(value):
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float, decimal.Decimal)):
        d = _decimal(str(value))
        return _number_key(d) if d is not None else ("float", repr(value))
    if isinstance(value, complex):
        return ("complex", repr(value))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__,
                tuple(_unify_strings(_value_key(v)) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted(
            ((_unify_strings(_value_key(k)), _unify_strings(_value_key(v)))
             for k, v in value.items()),
            key=repr)))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__,
                tuple(sorted((_unify_strings(_value_key(v)) for v in value),
                             key=repr)))
    return ("raw", repr(value))


# --------------------------------------------------------------------------
# gold
# --------------------------------------------------------------------------
def gold_key(gold, bench):
    """Canonical key for a dataset gold answer (already a clean literal).

    Routed through the same unification as a reply, so that a gold given as
    ``('Not matched!')`` and a reply written as ``Not matched!`` land on the
    same key.
    """
    if gold is None:
        return None
    text = str(gold).strip()
    if bench == "mbpp":
        return _literal_key(text)
    return norm_answer(text, bench)


def correct(answer, gold, bench):
    """True when the reply matches the gold under the benchmark's semantics."""
    ka = norm_answer(answer, bench)
    if ka is None:
        return False
    return ka == gold_key(gold, bench)


# --------------------------------------------------------------------------
# KEY (separate rule: never apply answer-only reductions such as the MMLU
# option collapse, and never delete punctuation that changes meaning)
# --------------------------------------------------------------------------
def norm_key(text):
    """Cosmetic lexical normalisation of the decisive-step field.

    This makes no semantic-equivalence claim; see Section 4.5 of the paper.
    """
    s = _strip_schema(text)
    if s is None:
        return None
    s = " ".join(s.split())
    if not s:
        return None
    return s.casefold()


# --------------------------------------------------------------------------
# schema parsing
# --------------------------------------------------------------------------
_ANSWER_FIELD = re.compile(r"(?:FINAL\s+)?ANSWER\s*[:：]\s*(.+)", re.I | re.M)
_KEY_FIELD = re.compile(r"KEY\s*[:：]\s*(.+)", re.I | re.M)


def parse_schema(text):
    """Return the last (ANSWER, KEY) pair in a reply.

    The v1 parser took the last match of each field independently, so a reply
    that mentioned ``ANSWER:`` twice but ``KEY:`` once could pair the wrong
    lines.  Here the last *block* is used: the KEY line that follows the last
    ANSWER line, or the last ANSWER line and the last KEY line when the order
    is reversed.
    """
    if not text:
        return (None, None)
    answers = list(_ANSWER_FIELD.finditer(text))
    keys = list(_KEY_FIELD.finditer(text))
    a = answers[-1].group(1).strip() if answers else None
    if a is not None:
        after = [k for k in keys if k.start() > answers[-1].end()]
        k = (after[0] if after else (keys[-1] if keys else None))
    else:
        k = keys[-1] if keys else None
    k = k.group(1).strip() if k is not None else None
    return (a, k)


# --------------------------------------------------------------------------
# classes
# --------------------------------------------------------------------------
def answer_class(answer, bench):
    """Verdict-latitude class: the comparison majority voting performs."""
    return norm_answer(answer, bench)


def semantic_class(answer, key, bench):
    """Answer+KEY lexical class.  A proxy, not an ideal semantic partition."""
    a = norm_answer(answer, bench)
    if a is None:
        return None
    k = norm_key(key)
    return (a, k)
