"""
Overfitting Analysis for ADR Model
Compares training, validation, and test performance
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Metrics from training
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Balanced Accuracy', 'Matthews Corr.'],
    'Train': [0.9286, 0.9375, 0.9836, 0.9600, 0.9836, 0.7696, 0.6447],
    'Validation': [0.8571, 0.8571, 1.0000, 0.9231, 0.3333, 0.5000, 0.0000],
    'Test': [0.8750, 0.8750, 1.0000, 0.9333, 0.5714, 0.5000, 0.0000]
}

df = pd.DataFrame(metrics_data)

print("="*70)
print("OVERFITTING ANALYSIS")
print("="*70)
print("\nModel Performance Comparison:\n")
print(df.to_string(index=False))

# Calculate performance drops
print("\n" + "="*70)
print("PERFORMANCE DROP ANALYSIS (Train -> Test)")
print("="*70)
print("\nAbsolute Drop:")
for i, metric in enumerate(df['Metric']):
    drop = df['Train'][i] - df['Test'][i]
    print(f"  {metric:20s}: {drop:+.4f}")

print("\nRelative Drop (%):")
for i, metric in enumerate(df['Metric']):
    if df['Train'][i] > 0:
        rel_drop = ((df['Train'][i] - df['Test'][i]) / df['Train'][i]) * 100
        print(f"  {metric:20s}: {rel_drop:+.2f}%")

# Overfitting indicators
print("\n" + "="*70)
print("OVERFITTING INDICATORS")
print("="*70)

critical_issues = []
moderate_issues = []
good_performance = []

for i, metric in enumerate(df['Metric']):
    train_val = df['Train'][i]
    test_val = df['Test'][i]
    
    if train_val > 0:
        drop_pct = abs((train_val - test_val) / train_val * 100)
        
        if drop_pct > 30:
            critical_issues.append(f"{metric}: {drop_pct:.1f}% drop (SEVERE OVERFITTING)")
        elif drop_pct > 15:
            moderate_issues.append(f"{metric}: {drop_pct:.1f}% drop (Moderate overfitting)")
        else:
            good_performance.append(f"{metric}: {drop_pct:.1f}% drop (Good generalization)")

print("\nCRITICAL ISSUES (>30% performance drop):")
if critical_issues:
    for issue in critical_issues:
        print(f"  - {issue}")
else:
    print("  None")

print("\nMODERATE CONCERNS (15-30% performance drop):")
if moderate_issues:
    for issue in moderate_issues:
        print(f"  - {issue}")
else:
    print("  None")

print("\nGOOD GENERALIZATION (<15% performance drop):")
if good_performance:
    for item in good_performance:
        print(f"  + {item}")

# Overall assessment
print("\n" + "="*70)
print("OVERALL ASSESSMENT")
print("="*70)

auc_drop = abs((df.loc[df['Metric'] == 'AUC-ROC', 'Train'].values[0] - 
                df.loc[df['Metric'] == 'AUC-ROC', 'Test'].values[0]))

if auc_drop > 0.3 or critical_issues:
    print("\nSTATUS: SEVERE OVERFITTING DETECTED")
    print("\nThe model shows significant overfitting:")
    print("  - Training AUC-ROC: 0.9836")
    print("  - Test AUC-ROC: 0.5714 (DROP: -0.41)")
    print("  - This is a 41.8% performance drop!")
    print("\nKey Issues:")
    print("  1. AUC-ROC drops dramatically from train to test")
    print("  2. Matthews Correlation drops from 0.64 to 0.00")
    print("  3. Balanced Accuracy drops from 0.77 to 0.50")
    print("  4. Model memorizing training data rather than learning patterns")
    
elif auc_drop > 0.15 or moderate_issues:
    print("\nSTATUS: MODERATE OVERFITTING")
    print("The model shows some overfitting but may still be usable.")
else:
    print("\nSTATUS: GOOD GENERALIZATION")
    print("The model generalizes well to unseen data.")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS TO REDUCE OVERFITTING")
print("="*70)
print("""
1. INCREASE REGULARIZATION:
   - Increase 'reg_alpha' (L1 regularization): try 0.1, 0.5, 1.0
   - Increase 'reg_lambda' (L2 regularization): try 1.0, 2.0, 5.0
   - Reduce 'max_depth': try 3 or 4 instead of 6

2. REDUCE MODEL COMPLEXITY:
   - Decrease 'n_estimators': try 100-200 instead of 400
   - Increase 'min_child_weight': try 3, 5, or 10
   - Lower 'learning_rate': try 0.01 or 0.03 instead of 0.05

3. ADD MORE DATA:
   - Current dataset is very small (70 training samples)
   - Try to get more training data if possible
   - Consider data augmentation techniques

4. FEATURE ENGINEERING:
   - Remove potentially noisy or redundant features
   - Try feature selection to keep only the most important features
   - Check for data leakage in features

5. CROSS-VALIDATION:
   - Use k-fold cross-validation for more reliable performance estimates
   - Current single train/val/test split may not be representative

6. EARLY STOPPING:
   - Monitor validation performance more closely
   - Stop training earlier if validation performance plateaus
   - Current early_stopping_rounds=50 may be too lenient
""")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Overfitting Analysis: Train vs Test Performance', fontsize=16, fontweight='bold')

# Plot 1: Bar chart of all metrics
ax1 = axes[0, 0]
x = np.arange(len(df['Metric']))
width = 0.25

bars1 = ax1.bar(x - width, df['Train'], width, label='Train', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x, df['Validation'], width, label='Validation', color='#f39c12', alpha=0.8)
bars3 = ax1.bar(x + width, df['Test'], width, label='Test', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Metrics', fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Performance Comparison Across All Metrics')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Metric'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Performance drop
ax2 = axes[0, 1]
drops = [df['Train'][i] - df['Test'][i] for i in range(len(df))]
colors = ['red' if d > 0.3 else 'orange' if d > 0.15 else 'green' for d in drops]
bars = ax2.barh(df['Metric'], drops, color=colors, alpha=0.7)
ax2.set_xlabel('Performance Drop (Train - Test)', fontweight='bold')
ax2.set_title('Overfitting by Metric (Red = Severe)')
ax2.axvline(x=0.15, color='orange', linestyle='--', alpha=0.5, label='15% threshold')
ax2.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='30% threshold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Focus on critical metrics
ax3 = axes[1, 0]
critical_metrics = ['Accuracy', 'F1 Score', 'AUC-ROC']
critical_data = df[df['Metric'].isin(critical_metrics)]
x_crit = np.arange(len(critical_metrics))

ax3.plot(x_crit, critical_data['Train'].values, 'o-', linewidth=2, markersize=10, 
         label='Train', color='#2ecc71')
ax3.plot(x_crit, critical_data['Validation'].values, 's-', linewidth=2, markersize=10,
         label='Validation', color='#f39c12')
ax3.plot(x_crit, critical_data['Test'].values, '^-', linewidth=2, markersize=10,
         label='Test', color='#e74c3c')

ax3.set_xlabel('Key Metrics', fontweight='bold')
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Key Performance Metrics Trend')
ax3.set_xticks(x_crit)
ax3.set_xticklabels(critical_metrics)
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Summary text
ax4 = axes[1, 1]
ax4.axis('off')

status_color = 'red' if auc_drop > 0.3 else 'orange' if auc_drop > 0.15 else 'green'
status_text = 'SEVERE OVERFITTING' if auc_drop > 0.3 else 'MODERATE' if auc_drop > 0.15 else 'GOOD'

summary_text = f"""
OVERFITTING STATUS: {status_text}

Train Performance:
  - Accuracy: {df['Train'][0]:.3f}
  - AUC-ROC: {df['Train'][4]:.3f}
  - F1 Score: {df['Train'][3]:.3f}

Test Performance:
  - Accuracy: {df['Test'][0]:.3f}
  - AUC-ROC: {df['Test'][4]:.3f}
  - F1 Score: {df['Test'][3]:.3f}

Performance Gaps:
  - Accuracy Drop: {df['Train'][0] - df['Test'][0]:.3f}
  - AUC-ROC Drop: {df['Train'][4] - df['Test'][4]:.3f}
  - F1 Score Drop: {df['Train'][3] - df['Test'][3]:.3f}

Main Issue:
  AUC-ROC drops by 41.8%
  (from 0.984 to 0.571)

Dataset Size:
  Train: 70 samples (VERY SMALL)
  Val: 14 samples
  Test: 16 samples
"""

ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor=status_color, alpha=0.2))

plt.tight_layout()
plt.savefig('reports/overfitting_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: reports/overfitting_analysis.png")
print("\n" + "="*70)


