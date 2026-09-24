"""Shared sampling layer for the v3 online experiments.

Design constraints, each responding to a specific defect the review found in the
v1 scripts:

  * Every call records the full raw reply, the finish reason, token counts, the
    model string, the request parameters and a config hash.  The v1 main
    sampling files kept only the parsed ``answer``/``key`` fields, so parse
    failures could not be diagnosed after the fact.
  * A run writes to a new file by default and refuses to append to an existing
    one unless ``resume=True`` is passed explicitly.  The v1 sampler appended
    silently while overwriting the summary, so re-running merged two samples.
  * Failures are counted and reported per category rather than dropped, and a
    run refuses to report statistics over calls it could not parse without
    saying how many those were.
  * Credentials are read from a dotenv file and are never logged, echoed, or
    written into any artifact.

Usage
-----
    from sampling import Sampler, schema_prompt

    s = Sampler(model="gpt-4.1-nano", env=".env")
    rec = s.complete(system=schema_prompt(), user=question)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import re
import time
from dataclasses import dataclass, field, asdict

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
RESULTS = V3 / "results"


# --------------------------------------------------------------------------
# prompts.  Kept in one place so a condition cannot silently change text.
# --------------------------------------------------------------------------
SYSTEM_FREE = (
    "Solve the question. Think step by step first, then end your reply with "
    "EXACTLY these two lines and nothing after them:\n"
    "ANSWER: <final answer, minimal form>\n"
    "KEY: <the single decisive step, <=10 words>"
)

SYSTEM_FREE_FORMAL = (
    "Solve the question. Think step by step first. Then state your final "
    "answer and the single decisive step, using exactly this format as the "
    "last two lines and nothing after them:\n"
    "ANSWER: <final answer, minimal form>\n"
    "KEY: <the single decisive step, <=10 words>"
)

SYSTEM_SCHEMA_ONLY = (
    "Reply with EXACTLY two lines and nothing else:\n"
    "ANSWER: <final answer, minimal form>\n"
    "KEY: <the single decisive step, <=10 words>"
)

SYSTEM_CODE = (
    "Write a Python function that solves the problem. Return ONLY the function "
    "definition inside a single fenced code block, with no explanation before "
    "or after it."
)

BYZ_FIXED = (
    "Your assigned role is to argue confidently and persuasively for the answer "
    "{w}, giving a plausible-sounding justification, regardless of what the "
    "correct answer is. Never admit doubt."
)

BYZ_ADAPTIVE = (
    "Your assigned role is to argue confidently and persuasively that the most "
    "popular answer offered by the other participants is wrong, and that a "
    "different value ({w}) is correct. Give a plausible-sounding "
    "justification. Never admit doubt."
)

JUDGE_PLAIN = (
    "Two replies to the same question are given. Decide whether they express "
    "the SAME solution: the same final answer reached by the same decisive "
    "step. Wording, language, and symbolic form do not matter; a different "
    "final answer always means DIFFERENT. Answer with exactly one word: "
    "SAME or DIFFERENT.\n\nReply A:\n{a}\n\nReply B:\n{b}"
)

JUDGE_COT = (
    "Two replies to the same question are given. First compare their final "
    "answers. If the final answers differ, the verdict is DIFFERENT. Only if "
    "the final answers agree should you compare the decisive steps, and the "
    "verdict is SAME only when those steps describe the same operation. "
    "Wording, language, and symbolic form do not matter.\n"
    "Think briefly, then end with exactly one line: VERDICT: SAME or "
    "VERDICT: DIFFERENT.\n\nReply A:\n{a}\n\nReply B:\n{b}"
)

JUDGE_FEWSHOT = (
    "Two replies to the same question are given. Decide whether they express "
    "the SAME solution: the same final answer reached by the same decisive "
    "step.\n\n"
    "Examples:\n"
    "A: ANSWER 7 / KEY factor the polynomial   B: ANSWER 7 / KEY use the "
    "quadratic formula  -> DIFFERENT (same answer, different method)\n"
    "A: ANSWER 72 / KEY add april and may sales   B: ANSWER 72 / KEY "
    "48 + 24 = 72  -> SAME (same step, different notation)\n"
    "A: ANSWER 12 / KEY count characters in the string   B: ANSWER 12 / KEY "
    "return len(str1)  -> SAME\n"
    "A: ANSWER 5 / KEY subtract the remainder   B: ANSWER 6 / KEY subtract "
    "the remainder  -> DIFFERENT (different answers)\n\n"
    "Answer with exactly one word: SAME or DIFFERENT.\n\nReply A:\n{a}\n\n"
    "Reply B:\n{b}"
)


def schema_prompt(free: bool = True) -> str:
    return SYSTEM_FREE if free else SYSTEM_SCHEMA_ONLY


# --------------------------------------------------------------------------
def load_env(path=None, required=("OPENAI_API_KEY",)):
    """Minimal dotenv reader.  Values are never printed or logged."""
    candidates = [path] if path else [HERE / ".env", ROOT / "exp" / ".env"]
    for cand in candidates:
        if cand and pathlib.Path(cand).exists():
            for line in pathlib.Path(cand).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            break
    return [k for k in required if not os.environ.get(k)]


# --------------------------------------------------------------------------
# Provider routing
# --------------------------------------------------------------------------
# The relay used for this project speaks the Anthropic messages API rather than
# OpenAI chat completions, and its edge rejects requests that carry no
# User-Agent with "Access denied by security policy (WAF)".  Both facts are
# handled here so the experiment scripts stay provider-agnostic.
ANTHROPIC_STYLE_PREFIXES = ("claude", "gpt-6", "gpt-5", "opus", "sonnet", "haiku")
DEFAULT_UA = "bft-experiment/1.0 (+research; python-urllib)"

#: which credential and endpoint each model family needs
PROVIDERS = {
    "relay": {"key": "AICODING_API_KEY", "base": "AICODING_BASE_URL",
              "path": "/v1/messages", "style": "anthropic"},
    "openai": {"key": "OPENAI_API_KEY", "base": "OPENAI_BASE_URL",
               "path": "/chat/completions", "style": "openai"},
    "deepseek": {"key": "DEEPSEEK_API_KEY", "base": "DEEPSEEK_BASE_URL",
                 "path": "/chat/completions", "style": "openai"},
}


def provider_for(model, provider=None):
    """Which provider serves ``model``.

    ``provider`` overrides the guess when a model name is ambiguous, e.g.
    requesting an OpenAI model through the relay.
    """
    if provider:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; "
                             f"choose from {sorted(PROVIDERS)}")
        return provider
    m = (model or "").lower()
    if any(m.startswith(p) for p in ANTHROPIC_STYLE_PREFIXES):
        return "relay"
    if m.startswith("deepseek"):
        return "deepseek"
    return "openai"



def config_hash(**kwargs) -> str:
    blob = json.dumps(kwargs, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
@dataclass
class Record:
    task_id: str
    model: str
    condition: str
    raw: str | None = None
    answer: str | None = None
    key: str | None = None
    code: str | None = None
    finish_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


class Sampler:
    """A recording wrapper over one chat endpoint, OpenAI- or Anthropic-style.

    The provider is chosen from the model name (see :func:`provider_for`), so
    the experiment scripts do not need to know which API a model speaks.
    Every call records the full raw reply, the finish reason, token counts,
    latency and a config hash; nothing is dropped on the floor.
    """

    def __init__(self, model, env=None, provider=None, temperature=1.0,
                 max_tokens=700, extra_body=None, no_think=False, timeout=600):
        self.provider = provider_for(model, provider)
        spec = PROVIDERS[self.provider]
        missing = load_env(env, required=(spec["key"],))
        if missing:
            raise SystemExit(
                f"missing credential {missing} for provider {self.provider!r}. "
                f"Put it in {HERE / '.env'} as KEY=value.  The file is never "
                f"read into any result artifact.")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base = (os.environ.get(spec["base"]) or "").rstrip("/")
        self.api_key = os.environ[spec["key"]]
        self.extra_body = dict(extra_body or {})
        if no_think:
            # Qwen hybrid models accept this to suppress the thinking channel.
            self.extra_body.setdefault("enable_thinking", False)
        self.usage = {"in": 0, "out": 0, "calls": 0, "errors": 0}
        self.last_error = None
        self.config = config_hash(model=model, provider=self.provider,
                                  temperature=temperature, max_tokens=max_tokens,
                                  extra_body=self.extra_body)

        self._client = None
        if spec["style"] == "openai":
            from openai import OpenAI
            kwargs = {"api_key": self.api_key, "timeout": timeout,
                      "default_headers": {"User-Agent": DEFAULT_UA}}
            if self.base:
                kwargs["base_url"] = self.base
            self._client = OpenAI(**kwargs)

    # ------------------------------------------------------------------
    def _call_openai(self, system, user):
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **({"extra_body": self.extra_body} if self.extra_body else {}),
        )
        choice = resp.choices[0]
        u = getattr(resp, "usage", None)
        return {
            "raw": choice.message.content or "",
            "finish_reason": getattr(choice, "finish_reason", None),
            "tokens_in": getattr(u, "prompt_tokens", 0) or 0,
            "tokens_out": getattr(u, "completion_tokens", 0) or 0,
        }

    def _call_anthropic(self, system, user):
        """Anthropic messages API over urllib.

        A User-Agent is mandatory: the relay's edge returns
        "Access denied by security policy (WAF)" for requests that omit one.
        """
        import urllib.request
        url = (self.base or "https://api.aicoding.sh") + "/v1/messages"
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        body.update(self.extra_body)
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": DEFAULT_UA,
                "Accept": "application/json",
            },
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        if not text:
            # a refusal or a non-text-only response must not look like an
            # empty-but-successful generation
            types = [b.get("type") for b in blocks if isinstance(b, dict)]
            raise RuntimeError(
                f"no text block in response (blocks={types}, "
                f"stop_reason={data.get('stop_reason')})")
        u = data.get("usage") or {}
        return {
            "raw": text,
            "finish_reason": data.get("stop_reason"),
            "tokens_in": u.get("input_tokens", 0) or 0,
            "tokens_out": u.get("output_tokens", 0) or 0,
        }

    # ------------------------------------------------------------------
    def complete(self, system, user, retries=3, backoff=1.6):
        """One completion, with the full reply retained."""
        from scoring import parse_schema
        call = self._call_anthropic if self.provider == "relay" else self._call_openai
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                out = call(system, user)
                text = out["raw"] or ""
                a, k = parse_schema(text)
                self.usage["calls"] += 1
                self.usage["in"] += out["tokens_in"]
                self.usage["out"] += out["tokens_out"]
                return {
                    "raw": text, "answer": a, "key": k,
                    "finish_reason": out["finish_reason"],
                    "tokens_in": out["tokens_in"],
                    "tokens_out": out["tokens_out"],
                    "latency_s": time.time() - t0,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                detail = f"{type(exc).__name__}: {exc}"
                if hasattr(exc, "read"):
                    try:
                        detail += " | " + exc.read().decode()[:200].replace("\n", " ")
                    except Exception:  # noqa: BLE001
                        pass
                last = detail
                self.last_error = detail
                self.usage["errors"] += 1
                if attempt < retries - 1:
                    time.sleep(backoff ** attempt)
        return {"raw": None, "answer": None, "key": None, "finish_reason": None,
                "tokens_in": 0, "tokens_out": 0,
                "latency_s": time.time() - t0, "error": last}


def extract_code(text):
    """Pull the first fenced code block, or the whole text when unfenced."""
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).rstrip()
    m = re.search(r"^(?:from|import|def)\s.*", text, re.S | re.M)
    return m.group(0).rstrip() if m else None


class Writer:
    """Append-only JSONL writer for a resumable run.

    A resumed run must not redo work it already has: re-issuing a completed
    job wastes money and, worse, appends a second copy of the same job, which
    inflates the per-task sample counts and therefore the estimated class
    distribution.  The writer is given a ``job_key`` function; records whose
    key is already on disk are skipped, and a re-written key raises instead of
    silently duplicating.
    """

    def __init__(self, path, resume=False, job_key=None, key_of=None):
        """``job_key`` keys records being written; ``key_of`` keys jobs.

        Both default to each other's role, and either may be supplied alone.
        Loading the already-recorded keys does not depend on them: a resumed
        run must know what is on disk even if it never writes again.
        """
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.job_key = job_key if job_key is not None else key_of
        self.key_of = key_of if key_of is not None else self.job_key
        self.seen = set()
        if self.path.exists():
            if not resume:
                raise SystemExit(
                    f"{self.path} already exists. Pass a new --out path, or "
                    f"--resume to continue it. Silently appending merges two "
                    f"runs and was a defect in the earlier pipeline.")
            self.records = []
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.records.append(rec)
                if self.job_key is not None:
                    try:
                        self.seen.add(self.job_key(rec))
                    except (KeyError, TypeError):
                        continue
            self.mode = "a"
        else:
            self.records = []
            self.mode = "w"
        self.fh = self.path.open(self.mode)
        self.skipped = 0

    def load(self, key_of=None):
        """Keys of every job already present, derived by ``key_of`` or the key."""
        kf = key_of if key_of is not None else self.job_key
        if kf is None:
            return set()
        out = set()
        for rec in self.records:
            try:
                out.add(kf(rec))
            except (KeyError, TypeError):
                continue
        return out

    def todo(self, jobs, key_of):
        """Filter ``jobs`` down to those not already recorded.

        ``key_of`` maps a job to the same tuple :meth:`write` derives from a
        record, so a resumed run performs only the outstanding work.  Getting
        this wrong is expensive in both money and correctness: re-running a
        completed job bills for it again and, if it were appended, would
        inflate that task's sample count.
        """
        out = []
        for job in jobs:
            if key_of(job) in self.seen:
                self.skipped += 1
            else:
                out.append(job)
        return out

    def write(self, obj):
        if self.job_key is not None:
            k = self.job_key(obj)
            if k in self.seen:
                raise RuntimeError(
                    f"refusing to write a duplicate job {k!r} to {self.path}; "
                    f"this would inflate the sample count for that task")
            self.seen.add(k)
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_json(path, obj):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    return p


def stratified_tasks(tasks, n_per_bench=None, seed=20260911):
    """Deterministic task subset, balanced across benchmarks."""
    import collections
    by = collections.defaultdict(list)
    for t in tasks:
        by[t["bench"]].append(t)
    rng = random.Random(seed)
    out = []
    for bench, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda r: r["id"])
        if n_per_bench and n_per_bench < len(rows):
            rows = rng.sample(rows, n_per_bench)
        out.extend(rows)
    return out
