"""生成 FetchPush 实验结果图表"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))

# ── 数据（来自 demo.py 实验结果）──
phases = ['Phase A\nBaseline\n(No Memory)', 'Phase B\nLearning\n(Writing)', 'Phase C\nRecall\n(Using Memory)']
success_rates = [42, 53, 67]
colors = ['#94a3b8', '#60a5fa', '#34d399']
edge_colors = ['#64748b', '#3b82f6', '#10b981']

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(phases, success_rates, color=colors, edgecolor=edge_colors,
              linewidth=2, width=0.6, zorder=3)

# 数值标注
for bar, rate in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{rate}%', ha='center', va='bottom', fontsize=22, fontweight='bold',
            color='#1e293b')

# +25% 标注箭头
ax.annotate('+25%', xy=(2, 67), xytext=(0.5, 75),
            fontsize=20, fontweight='bold', color='#059669',
            arrowprops=dict(arrowstyle='->', color='#059669', lw=2.5),
            ha='center')

# 样式
ax.set_ylim(0, 90)
ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold', color='#334155')
ax.set_title('FetchPush-v3: Memory-Guided Robot Pushing\n90 episodes per phase, pure CPU',
             fontsize=16, fontweight='bold', color='#0f172a', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cbd5e1')
ax.spines['bottom'].set_color('#cbd5e1')
ax.tick_params(colors='#64748b', labelsize=12)
ax.yaxis.grid(True, alpha=0.3, color='#94a3b8', linestyle='--', zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(os.path.join(_dir, 'fetchpush-results.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('✅ fetchpush-results.png saved')

# ── 第二张：跨环境泛化 ──
fig2, ax2 = plt.subplots(figsize=(8, 5))

envs = ['FetchPush\n(Training)', 'FetchSlide\n(Transfer)']
baseline = [42, 4]
with_mem = [67, 12]

x = np.arange(len(envs))
w = 0.3

b1 = ax2.bar(x - w/2, baseline, w, label='Without Memory', color='#94a3b8',
             edgecolor='#64748b', linewidth=1.5, zorder=3)
b2 = ax2.bar(x + w/2, with_mem, w, label='With Memory', color='#34d399',
             edgecolor='#10b981', linewidth=1.5, zorder=3)

for bars_group in [b1, b2]:
    for bar in bars_group:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{int(bar.get_height())}%', ha='center', va='bottom',
                fontsize=16, fontweight='bold', color='#1e293b')

ax2.annotate('+8%', xy=(1 + w/2, 12), xytext=(1.4, 25),
            fontsize=16, fontweight='bold', color='#059669',
            arrowprops=dict(arrowstyle='->', color='#059669', lw=2),
            ha='center')

ax2.set_ylim(0, 80)
ax2.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold', color='#334155')
ax2.set_title('Cross-Environment Transfer\nFetchPush → FetchSlide', fontsize=14, fontweight='bold',
              color='#0f172a', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(envs, fontsize=12)
ax2.legend(fontsize=11, loc='upper right')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#cbd5e1')
ax2.spines['bottom'].set_color('#cbd5e1')
ax2.tick_params(colors='#64748b')
ax2.yaxis.grid(True, alpha=0.3, color='#94a3b8', linestyle='--', zorder=0)
ax2.set_axisbelow(True)

fig2.tight_layout()
fig2.savefig(os.path.join(_dir, 'cross-env-transfer.png'), dpi=150, bbox_inches='tight',
             facecolor='white', edgecolor='none')
print('✅ cross-env-transfer.png saved')
