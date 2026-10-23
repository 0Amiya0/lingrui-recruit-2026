"""Parse evaluation logs, export CSV, and plot loss in a separate process."""
import argparse
import csv
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parent
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def parse_log(text):
    iterations, evaluation = {}, {}
    for line in text.splitlines():
        m = re.search(rf'iter (\d+): loss ({NUMBER})', line)
        if m:
            iterations[int(m[1])] = float(m[2])
        m = re.search(rf'step (\d+): train loss ({NUMBER}), val loss ({NUMBER})', line)
        if m:
            evaluation[int(m[1])] = (float(m[2]), float(m[3]))
    if not evaluation:
        raise ValueError('log has no evaluation records')
    return {'iterations': dict(sorted(iterations.items())), 'evaluation': dict(sorted(evaluation.items()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, default=ROOT / 'train_log.txt')
    p.add_argument('--output', type=Path, default=ROOT / 'loss_curve.png')
    p.add_argument('--csv', type=Path, default=ROOT / 'artifacts/verification/loss.csv')
    a = p.parse_args()
    try:
        data = parse_log(a.log.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as e:
        p.exit(2, f'error: {e}\n')
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['step', 'train_loss', 'val_loss'])
        for step, pair in data['evaluation'].items():
            w.writerow([step, *pair])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    it, ev = data['iterations'], data['evaluation']
    if it:
        ax.plot(list(it), list(it.values()), alpha=.3, label='training batch loss (dropout active)')
    ax.plot(list(ev), [v[0] for v in ev.values()], 'o-', label='train evaluation loss')
    ax.plot(list(ev), [v[1] for v in ev.values()], 's-', label='validation loss')
    ax.set(xlabel='iteration', ylabel='cross-entropy (nats/token)', title='nanoGPT character model')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=130)
    plt.close(fig)
    print(f'saved {a.output} and {a.csv}; final evaluation: {list(ev.items())[-1]}')


if __name__ == '__main__':
    main()
