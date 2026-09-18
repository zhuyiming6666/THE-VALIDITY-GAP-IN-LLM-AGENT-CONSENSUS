"""Build v5 offline analyses, updated Figures 2/3, and manuscript.

Run from llmbft: python3 -B v5/code/build_revision.py --compile
Online prompted-arm collection is separate and never launched by this build.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig-bft-v4')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import revision_analysis as analysis

V4 = Path(__file__).resolve().parents[1]
ROOT, PAPER = V4.parent, V4/'paper'
SOURCES = {}


def source(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def restore_figures():
    """Reuse original PDF-extracted figures, without adding new panels."""
    mapping = {'fig2_activation.png': 'v4fig2.png',
               'fig3_observability.png': 'v4fig3.png',
               'fig4_certification.png': 'v5_cert_original.png'}
    outputs = []
    for src, dst in mapping.items():
        src_path = source(PAPER/'translation/images'/src)
        dst_path = PAPER/'fig'/dst
        shutil.copyfile(src_path, dst_path)
        outputs.append(dst_path)
    return outputs


def instrument_figure(rev):
    """Same two original panels, with corrected rates and 95% task-cluster intervals."""
    rates = rev['instruments']['rates']
    keys = ['lexical', 'embedding', 'judge_nano', 'judge_mini']
    names = ['Lexical', 'Embedding', 'Judge 1', 'Judge 2']
    blue, red, green = '#32658c', '#b34f36', '#39836e'
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.fontsize': 7.5, 'figure.dpi': 220, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(4)
    mu = [rates[k]['all']['estimate'] for k in keys]
    lo = [rates[k]['all']['estimate'] - rates[k]['all']['ci95'][0] for k in keys]
    hi = [rates[k]['all']['ci95'][1] - rates[k]['all']['estimate'] for k in keys]
    ax[0].bar(x, mu, color=[red, green, blue, '#849eb3'], width=.6,
              yerr=[lo, hi], capsize=2.5, error_kw={'lw': .9, 'ecolor': '#333333'})
    for i, v in enumerate(mu):
        ax[0].text(i, v + hi[i] + .025, f'{v:.3f}', ha='center', fontsize=8)
    ax[0].set(xticks=x, xticklabels=names, ylim=(0, 1.14),
              ylabel='SAME rate', title='(a) Same pairs, different relations')
    for field, off, col, title in [
            ('same_answer', -.18, blue, 'Same answer'),
            ('different_answer', .18, red, 'Different answer')]:
        n = rates['lexical'][field]['n_pairs']
        assert all(rates[k][field]['n_pairs'] == n for k in keys)
        val = [rates[k][field]['estimate'] for k in keys]
        err = [[rates[k][field]['estimate'] - rates[k][field]['ci95'][0] for k in keys],
               [rates[k][field]['ci95'][1] - rates[k][field]['estimate'] for k in keys]]
        ax[1].bar(x+off, val, .36, color=col, label=f'{title} ({n})',
                  yerr=err, capsize=2, error_kw={'lw': .8, 'ecolor': '#333333'})
    ax[1].set(xticks=x, xticklabels=names, ylim=(0, 1.05),
              ylabel='Conditional SAME rate', title='(b) Dependence on final answers')
    ax[1].legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(w_pad=1.8)
    paths = [PAPER/'fig'/f'v4fig4.{ext}' for ext in ['png', 'pdf']]
    for path in paths:
        fig.savefig(path, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    return paths


def compile_pdf():
    commands = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'iclr-05.tex'],
                ['bibtex', 'iclr-05']]
    commands += [commands[0]]*2
    for i, cmd in enumerate(commands):
        with (V4/'results'/f'latex-pass-{i+1}.log').open('w') as log:
            subprocess.run(cmd, cwd=PAPER, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (PAPER/'iclr-05.log').read_text()
    for extra in range(3):
        if 'Label(s) may have changed' not in log:
            break
        with (V4/'results'/f'latex-stabilize-{extra+1}.log').open('w') as output:
            subprocess.run(commands[0], cwd=PAPER, stdout=output, stderr=subprocess.STDOUT, check=True)
        log = (PAPER/'iclr-05.log').read_text()
    assert 'undefined on input line' not in log and 'There were undefined references' not in log
    assert 'Label(s) may have changed' not in log
    print(next(s for s in log.splitlines() if 'Output written on' in s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    analysis.build()
    path = source(V4/'results/revision_analysis.json')
    rev = json.loads(path.read_text())
    for rel, digest in rev['sources_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    from complete_todos import main as complete_todos
    complete_todos()
    if (V4/'results/matched_prompted.samples.jsonl').exists():
        from analyse_matched_prompted import analyse
        analyse()
    outputs = restore_figures() + instrument_figure(rev)
    outputs += [PAPER/'fig'/name for name in ['v5_activation.pdf', 'v5_activation.png', 'v5_observability.pdf', 'v5_observability.png']]
    # Guard against accidentally reintroducing the withdrawn expanded tables.
    tex = (PAPER/'iclr-05.tex').read_text()
    assert r'\input{revision_' not in tex
    assert r'\section*{AI Use Statement}' not in tex
    # The old guard asserted that `\TODO{0.87x}` and `\TODO{counts table}` were
    # still present, i.e. that the scoped revision left them unfilled.  Those
    # markers and the other archived-value TODO markers are now filled, so the
    # guard is dropped rather than inverted.
    if args.compile:
        compile_pdf()
        outputs.append(PAPER/'iclr-05.pdf')
    for p in [Path(__file__), V4/'code/complete_todos.py', V4/'code/analyse_matched_prompted.py',
              V4/'code/run_matched_prompted.py', V4/'results/key_length_sensitivity.json', PAPER/'iclr-05.tex', PAPER/'refs.bib',
              PAPER/'iclr2027_conference.sty', PAPER/'iclr2027_conference.bst',
              PAPER/'fig/figure1-validity-gap/figure1-validity-gap.pdf']:
        source(p)
    for name in ['matched_prompted.samples.jsonl', 'matched_prompted_analysis.json']:
        if (V4/'results'/name).exists():
            source(V4/'results'/name)
    manifest = {'scope': 'v5 offline build including KEY-prefix sensitivity, revised Figures 2/3 and matched prompted analysis',
        'model_calls': 0, 'stored_candidate_executions': 0,
        'sources_sha256': SOURCES, 'raw_analysis_sources': rev['sources_sha256'],
        'outputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    (V4/'results/revision_build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Offline build complete; online sampling is a separate recorded step.')


if __name__ == '__main__':
    main()
