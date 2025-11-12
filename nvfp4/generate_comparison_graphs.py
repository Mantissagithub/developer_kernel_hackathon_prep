#!/usr/bin/env python3
"""
Generate comparison graphs for NVFP4 benchmarks across all tested GPUs
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
gpus = ['A100\n40GB', 'H100\n80GB', 'B200\n192GB']
tokens_per_sec_standard = [66.01, 112.59, 153.07]
tokens_per_sec_nvfp4 = [103.29, 163.05, 169.90]

accuracy_standard = [88.48, 88.63, 88.55]
accuracy_nvfp4 = [86.73, 87.49, 86.73]

# Calculate speedups
speedups = [nvfp4/std for nvfp4, std in zip(tokens_per_sec_nvfp4, tokens_per_sec_standard)]

# Set up the plotting style
plt.style.use('seaborn-v0_8-darkgrid')
colors_standard = '#2E86AB'  # Blue
colors_nvfp4 = '#A23B72'     # Purple/Magenta

# ==================== TOKENS PER SECOND COMPARISON ====================
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(gpus))
width = 0.35

bars1 = ax.bar(x - width/2, tokens_per_sec_standard, width,
               label='BF16', color=colors_standard, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, tokens_per_sec_nvfp4, width,
               label='NVFP4', color=colors_nvfp4, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add speedup annotations
for i, (x_pos, speedup) in enumerate(zip(x, speedups)):
    ax.text(x_pos, max(tokens_per_sec_standard[i], tokens_per_sec_nvfp4[i]) + 10,
            f'{speedup:.2f}x',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='green', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax.set_xlabel('GPU Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Tokens per Second', fontsize=14, fontweight='bold')
ax.set_title('NVFP4 Throughput Performance Comparison\nNVIDIA Nemotron-Nano-9B-v2',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=12)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

# Add horizontal reference lines
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=150, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('results/all_gpus_throughput_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated: results/all_gpus_throughput_comparison.png")
plt.close()

# ==================== GSM8K ACCURACY COMPARISON ====================
fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width/2, accuracy_standard, width,
               label='BF16', color=colors_standard, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, accuracy_nvfp4, width,
               label='NVFP4', color=colors_nvfp4, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add accuracy drop annotations
accuracy_drops = [std - nvfp4 for std, nvfp4 in zip(accuracy_standard, accuracy_nvfp4)]
for i, (x_pos, drop) in enumerate(zip(x, accuracy_drops)):
    ax.text(x_pos, accuracy_nvfp4[i] - 1.5,
            f'Δ {drop:.2f}%',
            ha='center', va='top', fontsize=10, fontweight='bold',
            color='red', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

ax.set_xlabel('GPU Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('NVFP4 GSM8K Accuracy Comparison\nNVIDIA Nemotron-Nano-9B-v2',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=12)
ax.legend(fontsize=12, loc='lower right')
ax.set_ylim([84, 90])  # Focus on relevant accuracy range
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

# Add reference line at 85% and 88%
ax.axhline(y=85, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='85% baseline')
ax.axhline(y=88, color='green', linestyle='--', alpha=0.5, linewidth=1, label='88% baseline')

plt.tight_layout()
plt.savefig('results/all_gpus_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated: results/all_gpus_accuracy_comparison.png")
plt.close()

# ==================== COMBINED VIEW ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Throughput subplot
bars1_1 = ax1.bar(x - width/2, tokens_per_sec_standard, width,
                  label='BF16', color=colors_standard, alpha=0.8, edgecolor='black', linewidth=1.5)
bars1_2 = ax1.bar(x + width/2, tokens_per_sec_nvfp4, width,
                  label='NVFP4', color=colors_nvfp4, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1_1, bars1_2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

for i, (x_pos, speedup) in enumerate(zip(x, speedups)):
    ax1.text(x_pos, max(tokens_per_sec_standard[i], tokens_per_sec_nvfp4[i]) + 8,
            f'{speedup:.2f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='green', bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3))

ax1.set_xlabel('GPU Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Tokens per Second', fontsize=12, fontweight='bold')
ax1.set_title('Throughput Performance', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(gpus, fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_axisbelow(True)

# Accuracy subplot
bars2_1 = ax2.bar(x - width/2, accuracy_standard, width,
                  label='BF16', color=colors_standard, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2_2 = ax2.bar(x + width/2, accuracy_nvfp4, width,
                  label='NVFP4', color=colors_nvfp4, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars2_1, bars2_2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

for i, (x_pos, drop) in enumerate(zip(x, accuracy_drops)):
    ax2.text(x_pos, accuracy_nvfp4[i] - 0.8,
            f'Δ {drop:.2f}%',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax2.set_xlabel('GPU Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('GSM8K Accuracy', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(gpus, fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim([84, 90])
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_axisbelow(True)

fig.suptitle('NVFP4 Performance vs BF16 - NVIDIA Nemotron-Nano-9B-v2',
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('results/all_gpus_combined_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated: results/all_gpus_combined_comparison.png")
plt.close()

print("\n✅ All comparison graphs generated successfully!")
print("   - all_gpus_throughput_comparison.png")
print("   - all_gpus_accuracy_comparison.png")
print("   - all_gpus_combined_comparison.png")

