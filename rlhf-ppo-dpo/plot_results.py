"""Plot PPO training curves from results/ppo_history.json.

Usage: python plot_results.py
Writes results/ppo_training.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values, window=10):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main():
    results_dir = Path('results')
    history = json.loads((results_dir / 'ppo_history.json').read_text())
    steps = list(range(1, len(history) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, label in [(axes[0], 'mean_reward', 'Mean sentiment reward'), (axes[1], 'kl', 'KL to reference (per response)')]:
        values = [h[key] for h in history]
        ax.plot(steps, values, color='#9bb8d9', linewidth=1, label='per step')
        ax.plot(steps, moving_average(values), color='#1f5fa8', linewidth=2, label='10-step average')
        ax.set_xlabel('PPO step')
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(results_dir / 'ppo_training.png', dpi=150)
    print('wrote', results_dir / 'ppo_training.png')


if __name__ == '__main__':
    main()
