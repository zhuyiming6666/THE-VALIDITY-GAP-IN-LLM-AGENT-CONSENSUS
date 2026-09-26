"""Brute-force checks of the formal statements in the paper (numbering as in the manuscript).

Proposition 1 (collision identity), Proposition 2 (vacuity), Proposition 3 (refinement,
margin examples, mixtures), Theorem 1 (exact certificate, relabelling guard), Proposition 4
(honest margin 2f), Corollary 5 (Hoeffding bound is valid), Remark 6 (threshold degeneracy),
and the n-f quorum remark (3f).  Everything is exact enumeration on small instances.

    python experiments/E5_certification/code/check_paper_theory.py
Writes E5_certification/results/paper_theory_checks.json.
"""
import itertools, json, math, random
from collections import Counter
from pathlib import Path

rng = random.Random(20260924)
checks = {}


def check(name):
    def deco(fn):
        try:
            fn(); checks[name] = "pass"
        except AssertionError as e:
            checks[name] = f"FAIL: {e}"
        return fn
    return deco


def rand_p(m):
    w = [rng.random() for _ in range(m)]; s = sum(w); return [x / s for x in w]


def A(p, h): return sum(x ** h for x in p)


def T(H):
    """Strict plurality class of histogram H (dict), or None on ties."""
    if not H or max(H.values()) == 0: return None
    top = max(H.values()); w = [c for c, v in H.items() if v == top]
    return w[0] if len(w) == 1 else None


def certify(N, f, e=0):
    for c, v in N.items():
        if v > f + 2 * e + max([N[d] for d in N if d != c] + [0]): return c
    return None


def deletions(N, f):
    keys = list(N)
    for dl in itertools.product(*[range(min(f, N[k]) + 1) for k in keys]):
        if sum(dl) <= f: yield {k: N[k] - d for k, d in zip(keys, dl)}


def relabellings(N, e):
    """All histograms reachable by moving at most e entries between classes of N."""
    seen = {tuple(sorted(N.items()))}; frontier = [dict(N)]
    for _ in range(e):
        nxt = []
        for H in frontier:
            for a in H:
                if H[a] == 0: continue
                for b in H:
                    if a == b: continue
                    G = dict(H); G[a] -= 1; G[b] += 1; key = tuple(sorted(G.items()))
                    if key not in seen: seen.add(key); nxt.append(G)
        frontier = nxt
    return [dict(k) for k in seen]


def hists(n, m):
    for comb in itertools.product(range(n + 1), repeat=m):
        if sum(comb) == n: yield {i: comb[i] for i in range(m)}


@check("Prop 1: Pr[U_h] = sum_c p(c)^h <= p_max^(h-1), equality iff uniform on support")
def _():
    for _ in range(200):
        m, h = rng.randint(1, 5), rng.randint(2, 6); p = rand_p(m)
        exact = sum(math.prod(p[c] for c in seq) for seq in itertools.product(range(m), repeat=h) if len(set(seq)) == 1)
        assert abs(exact - A(p, h)) < 1e-12
        assert A(p, h) <= max(p) ** (h - 1) + 1e-12
    for m in range(1, 5):
        assert abs(A([1 / m] * m, 4) - (1 / m) ** 3) < 1e-12
    assert A([0.5, 0.3, 0.2], 3) < 0.5 ** 2 - 1e-6
    p = [0.5, 0.3, 0.2]; H3 = math.log(sum(x ** 3 for x in p)) / (1 - 3)
    assert abs(A(p, 3) - math.exp(-2 * H3)) < 1e-12


@check("Prop 2(i): constant rule satisfies CV with probability 1 - A(h) + p(d)^h")
def _():
    for _ in range(100):
        m, h = rng.randint(2, 4), rng.randint(2, 5); p = rand_p(m); d = rng.randrange(m)
        ok = sum(math.prod(p[c] for c in seq) for seq in itertools.product(range(m), repeat=h)
                 if not (len(set(seq)) == 1 and seq[0] != d))
        assert abs(ok - (1 - A(p, h) + p[d] ** h)) < 1e-12


@check("Prop 3: refinement cannot raise A(h); margin examples; product law <= averaged law")
def _():
    for _ in range(200):
        p = rand_p(3); i = rng.randrange(3); s = rng.random()
        q = [x for j, x in enumerate(p) if j != i] + [p[i] * s, p[i] * (1 - s)]
        for h in range(1, 8): assert A(q, h) <= A(p, h) + 1e-12
    marg = lambda p: sorted(p)[-1] - sorted(p)[-2]
    assert abs(marg([0.5, 0.3, 0.2]) - 0.2) < 1e-12
    assert abs(marg([0.3, 0.2, 0.3, 0.2]) - 0.0) < 1e-12          # split the mode 0.5 -> 0.3 + 0.2
    assert abs(marg([0.5, 0.15, 0.15, 0.2]) - 0.3) < 1e-12        # split the runner-up 0.3 -> 0.15 + 0.15
    for _ in range(200):
        h = rng.randint(2, 5); ps = [rand_p(3) for _ in range(h)]
        prod = sum(math.prod(pi[c] for pi in ps) for c in range(3))
        bar = [sum(pi[c] for pi in ps) / h for c in range(3)]
        assert prod <= A(bar, h) + 1e-12


@check("Theorem 1: guard N_c > f + max_d N_d returns the unique element of K_f(N)")
def _():
    for n in range(1, 9):
        for N in hists(n, 3):
            for f in range(0, n):
                K = None
                for H in deletions(N, f):
                    t = T(H); K = {t} if K is None else K & {t}
                K = {x for x in K if x is not None}
                c = certify(N, f)
                assert (c is None and not K) or (K == {c}), (N, f, K, c)


@check("Theorem 1 (relabelling): guard N_c > f + 2e + max_d N_d is exact")
def _():
    for n in range(1, 8):
        for N in hists(n, 3):
            for f in range(0, 3):
                for e in range(0, 3):
                    K = None
                    for Np in deletions(N, f):
                        for H in relabellings(Np, e):
                            t = T(H); K = {t} if K is None else K & {t}
                    K = {x for x in K if x is not None}
                    c = certify(N, f, e)
                    assert (c is None and not K) or (K == {c}), (N, f, e, K, c)


@check("Proposition 4: c* certified for every append of <= f labels iff honest lead > 2f")
def _():
    for h in range(1, 8):
        for H in hists(h, 3):
            for f in range(0, 4):
                star = max(H, key=lambda c: (H[c], -c))
                lead = H[star] - max(H[d] for d in H if d != star)
                ok = True
                for B in itertools.product(range(4), repeat=f):  # classes 0..2 plus an unseen class 3
                    N = dict(H); N[3] = 0
                    for b in B: N[b] += 1
                    if certify(N, f) != star: ok = False; break
                assert ok == (lead > 2 * f), (H, f, lead, ok)


@check("Corollary: Hoeffding lower bound never exceeds the exact coverage")
def _():
    for _ in range(150):
        m = rng.randint(2, 4); p = sorted(rand_p(m), reverse=True)
        if p[0] - p[1] < 1e-3: continue
        h, f = rng.randint(3, 9), rng.randint(0, 2); delta = p[0] - p[1]
        exact = 0.0
        for H in hists(h, m):
            if H[0] - max(H[d] for d in H if d != 0) > 2 * f:
                exact += math.factorial(h) / math.prod(math.factorial(H[c]) for c in H) * math.prod(p[c] ** H[c] for c in H)
        if h * delta > 2 * f:
            bound = max(0.0, 1 - m * math.exp(-(h * delta - 2 * f) ** 2 / (2 * h)))  # m-1 = support size
            assert bound <= exact + 1e-12, (p, h, f, bound, exact)


@check("Remark (degeneracy): 3f+1<=n<=3f+2 needs honest unanimity; n<=3f gives zero; n>=3f+3 admits dissent")
def _():
    for f in range(0, 4):
        for n in range(f + 1, 3 * f + 6):
            h = n - f
            admits_dissent = any(H[0] - max(H[1], H[2]) > 2 * f and H[0] < h for H in hists(h, 3))
            unanimity_ok = h > 2 * f
            if n <= 3 * f: assert not unanimity_ok
            elif n <= 3 * f + 2: assert unanimity_ok and not admits_dissent, (n, f)
            else: assert admits_dissent, (n, f)


@check("Remark (n-f quorum): omitting f honest votes and appending f requires honest lead > 3f")
def _():
    for h in range(1, 9):
        for H in hists(h, 3):
            for f in range(0, 3):
                if f > h: continue
                star = max(H, key=lambda c: (H[c], -c))
                lead = H[star] - max(H[d] for d in H if d != star)
                ok = True
                for Hd in deletions(H, f):
                    if sum(Hd.values()) != h - f: continue
                    for B in itertools.product(range(3), repeat=f):
                        N = dict(Hd)
                        for b in B: N[b] += 1
                        if certify(N, f) != star: ok = False; break
                    if not ok: break
                assert ok == (lead > 3 * f), (H, f, lead, ok)


@check("Theorem 7 (dispersion-limited resilience): both exponential bounds hold against exact coverage; beta* <= 1/3, = iff Delta = 1")
def _():
    for p in ([0.8, 0.15, 0.05], [0.6, 0.3, 0.1], [0.9, 0.1], [0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [0.95, 0.04, 0.01]):
        d = p[0] - p[1]; m1 = len(p)          # m-1 = support size (observed support plus unseen class, minus one)
        for n in (10, 20, 40, 60):
            for beta in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                f = int(beta * n); h = n - f
                exact = 0.0
                for H in hists(h, len(p)):
                    if H[0] - max(H[c] for c in H if c != 0) > 2 * f:
                        exact += math.factorial(h) / math.prod(math.factorial(H[c]) for c in H) * math.prod(p[c] ** H[c] for c in H)
                if h * d > 2 * f:
                    assert exact >= 1 - m1 * math.exp(-h / 2 * (d - 2 * f / h) ** 2) - 1e-12, (p, n, beta)
                if h * d < 2 * f:
                    assert exact <= math.exp(-h / 2 * (2 * f / h - d) ** 2) + 1e-12, (p, n, beta)
    for d in [i / 100 for i in range(101)]:
        b = d / (2 + d); assert b <= 1 / 3 + 1e-12 and (abs(b - 1 / 3) < 1e-12) == (d == 1.0)


@check("Proposition (shared context): A(h) = E_Z sum_c p_Z(c)^h >= sum_c p(c)^h; P_n -> Pr_Z[beta*(Delta_Z) > beta]")
def _():
    for _ in range(300):
        k = rng.randint(2, 4); w = rand_p(k); laws = [rand_p(3) for _ in range(k)]; h = rng.randint(2, 8)
        marg = [sum(w[z] * laws[z][c] for z in range(k)) for c in range(3)]
        assert sum(w[z] * A(laws[z], h) for z in range(k)) >= A(marg, h) - 1e-12
    # convergence of the optimal coverage under a two-context mixture, exact enumeration
    laws = [[0.9, 0.08, 0.02], [0.55, 0.35, 0.10]]; w = [0.6, 0.4]; beta = 0.15   # beta* = 0.29 and 0.09
    def cov(p, n):
        f = int(beta * n); h = n - f; tot = 0.0
        for H in hists(h, 3):
            if H[0] - max(H[1], H[2]) > 2 * f:
                tot += math.factorial(h) / math.prod(math.factorial(H[c]) for c in H) * math.prod(p[c] ** H[c] for c in H)
        return tot
    limit = sum(wz for wz, p in zip(w, laws) if (p[0] - p[1]) / (2 + p[0] - p[1]) > beta)
    gaps = [abs(sum(wz * cov(p, n) for wz, p in zip(w, laws)) - limit) for n in (20, 60, 120)]
    assert gaps[-1] < 0.02 and gaps[-1] <= gaps[0], gaps


out = Path(__file__).resolve().parents[1] / "results" / "paper_theory_checks.json"
out.write_text(json.dumps(checks, indent=1))
for k, v in checks.items(): print(f"{v:5}  {k}")
