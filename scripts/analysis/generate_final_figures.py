"""
Generate final comparison figures.
Compares CLIP vs CNN Baseline results.
"""

import os
import sys

# Change to project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_root)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Load results
with open('results/clip_results/complete_results.json') as f:
    clip_results = json.load(f)

with open('results/cnn_results/cnn_baseline_results.json') as f:
    baseline_results = json.load(f)

# Output directory
output_dir = Path('results/figures')
output_dir.mkdir(exist_ok=True)

# Extract data
subjects = list(range(1, 33))

# CLIP data
clip_valence_f1 = [clip_results['valence']['per_subject'][str(s)]['f1_score'] for s in subjects]
clip_arousal_f1 = [clip_results['arousal']['per_subject'][str(s)]['f1_score'] for s in subjects]
clip_valence_acc = [clip_results['valence']['per_subject'][str(s)]['accuracy'] for s in subjects]
clip_arousal_acc = [clip_results['arousal']['per_subject'][str(s)]['accuracy'] for s in subjects]

# Baseline data
baseline_valence_f1 = [baseline_results['valence']['per_subject'][str(s)]['f1_score'] for s in subjects]
baseline_arousal_f1 = [baseline_results['arousal']['per_subject'][str(s)]['f1_score'] for s in subjects]
baseline_valence_acc = [baseline_results['valence']['per_subject'][str(s)]['accuracy'] for s in subjects]
baseline_arousal_acc = [baseline_results['arousal']['per_subject'][str(s)]['accuracy'] for s in subjects]

# Summary statistics
clip_val_summary = clip_results['valence']['summary']
clip_aro_summary = clip_results['arousal']['summary']
baseline_val_summary = baseline_results['valence']['summary']
baseline_aro_summary = baseline_results['arousal']['summary']

print("=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print("\nVALENCE:")
print(f"  CLIP:     F1={clip_val_summary['mean_f1']:.3f}±{clip_val_summary['std_f1']:.3f}, Acc={clip_val_summary['mean_accuracy']:.3f}")
print(f"  Baseline: F1={baseline_val_summary['mean_f1']:.3f}±{baseline_val_summary['std_f1']:.3f}, Acc={baseline_val_summary['mean_accuracy']:.3f}")
print(f"  CLIP improvement: {((clip_val_summary['mean_f1']/baseline_val_summary['mean_f1'])-1)*100:.1f}%")

print("\nAROUSAL:")
print(f"  CLIP:     F1={clip_aro_summary['mean_f1']:.3f}±{clip_aro_summary['std_f1']:.3f}, Acc={clip_aro_summary['mean_accuracy']:.3f}")
print(f"  Baseline: F1={baseline_aro_summary['mean_f1']:.3f}±{baseline_aro_summary['std_f1']:.3f}, Acc={baseline_aro_summary['mean_accuracy']:.3f}")
print(f"  CLIP improvement: {((clip_aro_summary['mean_f1']/baseline_aro_summary['mean_f1'])-1)*100:.1f}%")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'clip': '#2196F3', 'baseline': '#FF9800'}

# Figure 1: F1 Score Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Valence
x = np.arange(2)
width = 0.35
vals_clip = [clip_val_summary['mean_f1'], clip_aro_summary['mean_f1']]
vals_base = [baseline_val_summary['mean_f1'], baseline_aro_summary['mean_f1']]
errs_clip = [clip_val_summary['std_f1'], clip_aro_summary['std_f1']]
errs_base = [baseline_val_summary['std_f1'], baseline_aro_summary['std_f1']]

bars1 = axes[0].bar(x - width/2, vals_base, width, label='CNN Baseline', color=colors['baseline'], yerr=errs_base, capsize=5)
bars2 = axes[0].bar(x + width/2, vals_clip, width, label='CLIP (Ours)', color=colors['clip'], yerr=errs_clip, capsize=5)

axes[0].set_ylabel('F1 Score', fontsize=12)
axes[0].set_title('Model Comparison: F1 Score', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Valence', 'Arousal'])
axes[0].legend()
axes[0].set_ylim(0, 0.6)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

# Add value labels
for bar, val in zip(bars1, vals_base):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, vals_clip):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Accuracy comparison
accs_clip = [clip_val_summary['mean_accuracy'], clip_aro_summary['mean_accuracy']]
accs_base = [baseline_val_summary['mean_accuracy'], baseline_aro_summary['mean_accuracy']]
accs_std_clip = [clip_val_summary['std_accuracy'], clip_aro_summary['std_accuracy']]
accs_std_base = [baseline_val_summary['std_accuracy'], baseline_aro_summary['std_accuracy']]

bars3 = axes[1].bar(x - width/2, accs_base, width, label='CNN Baseline', color=colors['baseline'], yerr=accs_std_base, capsize=5)
bars4 = axes[1].bar(x + width/2, accs_clip, width, label='CLIP (Ours)', color=colors['clip'], yerr=accs_std_clip, capsize=5)

axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Valence', 'Arousal'])
axes[1].legend()
axes[1].set_ylim(0, 0.6)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

for bar, val in zip(bars3, accs_base):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars4, accs_clip):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {output_dir / 'model_comparison_summary.png'}")

# Figure 2: Per-subject F1 Comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

x = np.arange(len(subjects))
width = 0.35

# Valence per subject
axes[0].bar(x - width/2, baseline_valence_f1, width, label='CNN Baseline', color=colors['baseline'], alpha=0.8)
axes[0].bar(x + width/2, clip_valence_f1, width, label='CLIP (Ours)', color=colors['clip'], alpha=0.8)
axes[0].axhline(y=np.mean(baseline_valence_f1), color=colors['baseline'], linestyle='--', alpha=0.7)
axes[0].axhline(y=np.mean(clip_valence_f1), color=colors['clip'], linestyle='--', alpha=0.7)
axes[0].set_ylabel('F1 Score', fontsize=12)
axes[0].set_title('Valence Classification: Per-Subject F1 Scores (LOSO CV)', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'S{s}' for s in subjects], rotation=45, ha='right')
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, 0.8)

# Arousal per subject
axes[1].bar(x - width/2, baseline_arousal_f1, width, label='CNN Baseline', color=colors['baseline'], alpha=0.8)
axes[1].bar(x + width/2, clip_arousal_f1, width, label='CLIP (Ours)', color=colors['clip'], alpha=0.8)
axes[1].axhline(y=np.mean(baseline_arousal_f1), color=colors['baseline'], linestyle='--', alpha=0.7)
axes[1].axhline(y=np.mean(clip_arousal_f1), color=colors['clip'], linestyle='--', alpha=0.7)
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('Arousal Classification: Per-Subject F1 Scores (LOSO CV)', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'S{s}' for s in subjects], rotation=45, ha='right')
axes[1].legend(loc='upper right')
axes[1].set_ylim(0, 0.8)

plt.tight_layout()
plt.savefig(output_dir / 'per_subject_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'per_subject_f1_comparison.png'}")

# Figure 3: Box Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Valence box plot
bp1 = axes[0].boxplot([baseline_valence_f1, clip_valence_f1],
                       labels=['CNN Baseline', 'CLIP (Ours)'],
                       patch_artist=True)
bp1['boxes'][0].set_facecolor(colors['baseline'])
bp1['boxes'][1].set_facecolor(colors['clip'])
axes[0].set_ylabel('F1 Score', fontsize=12)
axes[0].set_title('Valence: F1 Distribution Across Subjects', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 0.8)

# Arousal box plot
bp2 = axes[1].boxplot([baseline_arousal_f1, clip_arousal_f1],
                       labels=['CNN Baseline', 'CLIP (Ours)'],
                       patch_artist=True)
bp2['boxes'][0].set_facecolor(colors['baseline'])
bp2['boxes'][1].set_facecolor(colors['clip'])
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('Arousal: F1 Distribution Across Subjects', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 0.8)

plt.tight_layout()
plt.savefig(output_dir / 'f1_distribution_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'f1_distribution_boxplot.png'}")

# Figure 4: Summary Statistics Table as Image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table_data = [
    ['Metric', 'CNN Baseline', 'CLIP (Ours)', 'Improvement'],
    ['Valence F1', f"{baseline_val_summary['mean_f1']:.3f}±{baseline_val_summary['std_f1']:.3f}",
     f"{clip_val_summary['mean_f1']:.3f}±{clip_val_summary['std_f1']:.3f}",
     f"+{((clip_val_summary['mean_f1']/baseline_val_summary['mean_f1'])-1)*100:.1f}%"],
    ['Valence Accuracy', f"{baseline_val_summary['mean_accuracy']:.3f}",
     f"{clip_val_summary['mean_accuracy']:.3f}", '-'],
    ['Arousal F1', f"{baseline_aro_summary['mean_f1']:.3f}±{baseline_aro_summary['std_f1']:.3f}",
     f"{clip_aro_summary['mean_f1']:.3f}±{clip_aro_summary['std_f1']:.3f}",
     f"+{((clip_aro_summary['mean_f1']/baseline_aro_summary['mean_f1'])-1)*100:.1f}%"],
    ['Arousal Accuracy', f"{baseline_aro_summary['mean_accuracy']:.3f}",
     f"{clip_aro_summary['mean_accuracy']:.3f}", '-'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center',
                 colColours=['#E8E8E8']*4)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color the header
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Color improvement column
for i in range(1, 5):
    if '+' in str(table_data[i][3]):
        table[(i, 3)].set_facecolor('#C6EFCE')

plt.title('CNN vs CLIP Performance Comparison (DEAP Dataset, 32 Subjects, LOSO CV)',
          fontsize=14, fontweight='bold', pad=20)
plt.savefig(output_dir / 'summary_statistics_table.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'summary_statistics_table.png'}")

# Figure 5: Radar/Spider Chart for Multi-metric comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))

categories = ['F1 Score', 'Precision', 'Recall', 'Balanced\nAccuracy']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Valence
val_baseline = [baseline_val_summary['mean_f1'], baseline_val_summary['mean_precision'],
                baseline_val_summary['mean_recall'], baseline_val_summary['mean_balanced_accuracy']]
val_clip = [clip_val_summary['mean_f1'],
            np.mean([clip_results['valence']['per_subject'][str(s)].get('precision', 0) for s in subjects]),
            np.mean([clip_results['valence']['per_subject'][str(s)].get('recall', 0) for s in subjects]),
            clip_val_summary['mean_balanced_accuracy']]

val_baseline += val_baseline[:1]
val_clip += val_clip[:1]

axes[0].plot(angles, val_baseline, 'o-', linewidth=2, label='CNN Baseline', color=colors['baseline'])
axes[0].fill(angles, val_baseline, alpha=0.25, color=colors['baseline'])
axes[0].plot(angles, val_clip, 'o-', linewidth=2, label='CLIP (Ours)', color=colors['clip'])
axes[0].fill(angles, val_clip, alpha=0.25, color=colors['clip'])
axes[0].set_xticks(angles[:-1])
axes[0].set_xticklabels(categories)
axes[0].set_ylim(0, 1)
axes[0].set_title('Valence Classification', fontsize=14, fontweight='bold', pad=20)
axes[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# Arousal
aro_baseline = [baseline_aro_summary['mean_f1'], baseline_aro_summary['mean_precision'],
                baseline_aro_summary['mean_recall'], baseline_aro_summary['mean_balanced_accuracy']]
aro_clip = [clip_aro_summary['mean_f1'],
            np.mean([clip_results['arousal']['per_subject'][str(s)].get('precision', 0) for s in subjects]),
            np.mean([clip_results['arousal']['per_subject'][str(s)].get('recall', 0) for s in subjects]),
            clip_aro_summary['mean_balanced_accuracy']]

aro_baseline += aro_baseline[:1]
aro_clip += aro_clip[:1]

axes[1].plot(angles, aro_baseline, 'o-', linewidth=2, label='CNN Baseline', color=colors['baseline'])
axes[1].fill(angles, aro_baseline, alpha=0.25, color=colors['baseline'])
axes[1].plot(angles, aro_clip, 'o-', linewidth=2, label='CLIP (Ours)', color=colors['clip'])
axes[1].fill(angles, aro_clip, alpha=0.25, color=colors['clip'])
axes[1].set_xticks(angles[:-1])
axes[1].set_xticklabels(categories)
axes[1].set_ylim(0, 1)
axes[1].set_title('Arousal Classification', fontsize=14, fontweight='bold', pad=20)
axes[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))

plt.tight_layout()
plt.savefig(output_dir / 'multi_metric_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'multi_metric_radar.png'}")

# Save final summary to JSON
final_summary = {
    'comparison': {
        'valence': {
            'baseline_f1': baseline_val_summary['mean_f1'],
            'baseline_f1_std': baseline_val_summary['std_f1'],
            'clip_f1': clip_val_summary['mean_f1'],
            'clip_f1_std': clip_val_summary['std_f1'],
            'improvement_percent': ((clip_val_summary['mean_f1']/baseline_val_summary['mean_f1'])-1)*100
        },
        'arousal': {
            'baseline_f1': baseline_aro_summary['mean_f1'],
            'baseline_f1_std': baseline_aro_summary['std_f1'],
            'clip_f1': clip_aro_summary['mean_f1'],
            'clip_f1_std': clip_aro_summary['std_f1'],
            'improvement_percent': ((clip_aro_summary['mean_f1']/baseline_aro_summary['mean_f1'])-1)*100
        }
    },
    'methodology': {
        'baseline': 'CNN (3 conv blocks, BatchNorm, class_weight=balanced)',
        'clip': 'CLIP ViT-B/32 + MLP with focal loss, SMOTE, threshold optimization',
        'evaluation': 'Leave-One-Subject-Out Cross-Validation (32 subjects)',
        'dataset': 'DEAP (32 subjects, 40 trials each)'
    }
}

with open(output_dir / 'final_comparison_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)
print(f"Saved: {output_dir / 'final_comparison_summary.json'}")

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
