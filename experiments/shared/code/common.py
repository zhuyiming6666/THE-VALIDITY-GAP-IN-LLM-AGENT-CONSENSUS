"""Shared normalisation / parsing so every script partitions outputs identically."""
import re

_NUM = r"-?\d+(?:\.\d+)?"


def norm(s):
    """Canonical form of a short answer.

    Steps: strip markdown / schema residue ('**', 'ANSWER:'), quotes, '$';
    an MMLU-style option ('B', 'B.', 'B. False, False') maps to its letter;
    a number with a trailing unit / percent / word maps to the number; tuples
    and code literals have quote style unified.  Known limitation: symbolic
    forms such as '9*sqrt(3)' vs '9 * math.sqrt(3)' are not unified.
    """
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"^(?:\*\*|`|#)+\s*", "", s)              # leading markdown
    s = re.sub(r"^(?:final\s+)?answer\s*[:：]\s*", "", s, flags=re.I)   # schema residue
    s = re.sub(r"(?:\*\*|`)+$", "", s).strip()           # trailing markdown
    s = s.lower().rstrip(".").replace(",", "").replace("$", "").strip("'\" ")
    m = re.fullmatch(r"([a-d])(?:[.)]\s*.*)?", s)          # multiple-choice letter
    if m:
        return m.group(1)
    m = re.fullmatch(rf"({_NUM})(?:\s*(?:%|[a-z][a-z .\-/]*))?", s)
    if m:
        return str(float(m.group(1)))
    s = s.replace('"', "'")                                # unify quote style
    return " ".join(s.split())


def parse(text):
    """Last ANSWER:/KEY: pair in a reply (reasoning may precede the schema)."""
    a = re.findall(r"(?:FINAL\s+)?ANSWER:\s*(.+)", text or "", re.I)
    k = re.findall(r"KEY:\s*(.+)", text or "", re.I)
    return ((a[-1].strip() if a else None), (k[-1].strip() if k else None))


if __name__ == "__main__":
    for t in ["72", "72 clips", "** 2", "ANSWER: 6", "$6", "B", "B. False, False", "C. True, False",
              '(0, 7, "clearly")', "(0, 7, 'clearly')", "** pythonprogram123", "Not matched!",
              "9*√3", "9 * math.sqrt(3)", "[1, 2]", "2 and 3", "a", "abc"]:
        print(f"{t!r:>24} -> {norm(t)!r}")
