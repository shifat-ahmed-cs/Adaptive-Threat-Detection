"""
Comprehensive Visualization Script for Thesis Chapter 5 & 6
Generates all publication-ready figures for:
  - Fusion method comparison (ROC, Radar, Confusion Matrices, PR Curves)
  - Adversarial robustness analysis (Heatmap, Recall Bar Chart)
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, average_precision_score
)

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))
from models import DualBranchModel

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data' / 'custom_dataset'
CKPT_DIR    = ROOT / 'checkpoints'
FIG_DIR     = ROOT / 'thesis_materials' / 'figures' / 'generated'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    'concat':         '#2563EB',   # blue
    'add':            '#16A34A',   # green
    'attention':      '#DC2626',   # red
    'gated':          '#7C3AED',   # purple
    'baseline':       '#1E40AF',   # dark blue
    'random_insert':  '#059669',   # teal
    'noise_append':   '#D97706',   # amber
    'random_replace': '#BE185D',   # pink
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
})


# ─────────────────────────────────────────────────────────────────────────────
# Utility: load model + run inference
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device='cpu'):
    ck   = torch.load(str(ckpt_path), map_location=device)
    mc   = ck.get('model_config', {})
    meta = ck.get('metadata', {})
    model = DualBranchModel(
        cnn_config ={'input_channels':1,'num_classes':2,
                     'feature_dim':mc.get('cnn_feature_dim',256)},
        lstm_config={'vocab_size':meta['vocab_size'],
                     'embedding_dim':mc.get('embedding_dim',100),
                     'hidden_size':mc.get('lstm_hidden_size',128),
                     'num_layers':mc.get('lstm_num_layers',2),
                     'feature_dim':mc.get('lstm_feature_dim',256),
                     'bidirectional':True},
        num_classes=2,
        fusion_method=mc.get('fusion_method','concat')
    )
    if ck.get('is_expanded', False):
        model.expand_capacity()
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model


def get_probs_preds(ckpt_path, cnn_arr, lstm_arr, labels_arr, device='cpu'):
    model  = load_model(ckpt_path, device)
    probs, preds = [], []
    bs = 32
    with torch.no_grad():
        for s in range(0, len(labels_arr), bs):
            cb = torch.FloatTensor(cnn_arr[s:s+bs]) / 255.0
            lb = torch.LongTensor(lstm_arr[s:s+bs])
            out = model(cb, lb)
            p   = torch.softmax(out, dim=1)[:,1].numpy()
            probs.extend(p.tolist())
            preds.extend((p >= 0.5).astype(int).tolist())
    del model
    return np.array(probs), np.array(preds), labels_arr


# ─────────────────────────────────────────────────────────────────────────────
# Load data once
# ─────────────────────────────────────────────────────────────────────────────
print("Loading test data...")
test_cnn    = np.load(DATA_DIR / 'test_cnn.npy')
test_lstm   = np.load(DATA_DIR / 'test_lstm.npy')
test_labels = np.load(DATA_DIR / 'test_labels.npy')

FUSION_CKPTS = {
    'Concat':    CKPT_DIR / 'baseline_concat.pth',
    'Add':       CKPT_DIR / 'best_add_norm.pth',
    'Attention': CKPT_DIR / 'best_attention_norm.pth',
    'Gated':     CKPT_DIR / 'best_gated_norm.pth',
}
MUTATION_CKPTS = {
    'Baseline':        CKPT_DIR / 'baseline_concat.pth',
    'Random Insert':   CKPT_DIR / 'mutation_random_insert.pth',
    'Noise Append':    CKPT_DIR / 'mutation_noise_append.pth',
    'Random Replace':  CKPT_DIR / 'mutation_random_replace.pth',
}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — ROC Curves: Fusion Methods
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] ROC Curves — Fusion Methods...")
fig, ax = plt.subplots(figsize=(7, 6))
colors  = list(PALETTE.values())[:4]

for (name, ckpt), col in zip(FUSION_CKPTS.items(), colors):
    probs, _, labels = get_probs_preds(ckpt, test_cnn, test_lstm, test_labels)
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name}  (AUC = {roc_auc:.4f})', linewidth=2, color=col)

ax.plot([0,1],[0,1],'--', color='grey', linewidth=1, label='Random Classifier')
ax.fill_between([0,1],[0,1], alpha=0.05, color='grey')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Fusion Method Comparison')
ax.legend(loc='lower right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_roc_fusion.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig1_roc_fusion.png', bbox_inches='tight')
plt.close()
print("   Saved fig1_roc_fusion.pdf/.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Radar / Spider Chart: Fusion Methods
# ─────────────────────────────────────────────────────────────────────────────
print("[2/7] Radar Chart — Fusion Methods...")

EVAL_FILES = {
    'Concat':    CKPT_DIR / 'evaluation_results_concat.json',
    'Add':       CKPT_DIR / 'evaluation_results_add.json',
    'Attention': CKPT_DIR / 'evaluation_results_attention.json',
    'Gated':     CKPT_DIR / 'evaluation_results_gated.json',
}

metric_keys  = ['accuracy','precision','recall','f1_score','auc']
metric_labels= ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
N = len(metric_keys)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, size=11)
ax.set_ylim(0.88, 1.0)
ax.set_yticks([0.90, 0.93, 0.96, 0.99])
ax.set_yticklabels(['90%','93%','96%','99%'], size=8)

for (name, jfile), col in zip(EVAL_FILES.items(), colors):
    with open(jfile) as f: d = json.load(f)
    m   = d['test_metrics']
    vals= [m[k] for k in metric_keys] + [m[metric_keys[0]]]
    ax.plot(angles, vals, 'o-', linewidth=2, color=col, label=name)
    ax.fill(angles, vals, alpha=0.07, color=col)

ax.set_title('Performance Profile — Fusion Method Comparison\n', size=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_radar_fusion.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig2_radar_fusion.png', bbox_inches='tight')
plt.close()
print("   Saved fig2_radar_fusion.pdf/.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Confusion Matrix Heatmaps: All Fusion Methods
# ─────────────────────────────────────────────────────────────────────────────
print("[3/7] Confusion Matrix Heatmaps — Fusion Methods...")

fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
axes = axes.flatten()
for ax, (name, jfile) in zip(axes, EVAL_FILES.items()):
    with open(jfile) as f: d = json.load(f)
    m  = d['test_metrics']
    cm = np.array([[m['true_negatives'],  m['false_positives']],
                   [m['false_negatives'], m['true_positives']]])
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{name} Fusion', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Benign','Malware']); ax.set_yticklabels(['Benign','Malware'])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}',
                    ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > thresh else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Confusion Matrices — Fusion Method Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_confusion_fusion.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig3_confusion_fusion.png', bbox_inches='tight')
plt.close()
print("   Saved fig3_confusion_fusion.pdf/.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Precision-Recall Curves: Fusion Methods
# ─────────────────────────────────────────────────────────────────────────────
print("[4/7] Precision-Recall Curves — Fusion Methods...")

fig, ax = plt.subplots(figsize=(7, 6))
for (name, ckpt), col in zip(FUSION_CKPTS.items(), colors):
    probs, _, labels = get_probs_preds(ckpt, test_cnn, test_lstm, test_labels)
    prec, rec, _     = precision_recall_curve(labels, probs)
    ap               = average_precision_score(labels, probs)
    ax.plot(rec, prec, label=f'{name}  (AP = {ap:.4f})', linewidth=2, color=col)

baseline_ap = np.sum(test_labels) / len(test_labels)
ax.axhline(y=baseline_ap, linestyle='--', color='grey', linewidth=1,
           label=f'No-Skill Baseline ({baseline_ap:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Fusion Method Comparison')
ax.legend(loc='lower left')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_pr_fusion.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig4_pr_fusion.png', bbox_inches='tight')
plt.close()
print("   Saved fig4_pr_fusion.pdf/.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Robustness Heatmap: 4×4 Accuracy Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("[5/7] Robustness Heatmap...")

with open(CKPT_DIR / 'robustness_results.json') as f:
    rob = json.load(f)

model_names   = ['Baseline','Random Insert','Noise Append','Random Replace']
dataset_names = ['Base','Random Insert','Noise Append','Random Replace']
model_keys    = list(rob.keys())
dataset_keys  = ['base','random_insert','noise_append','random_replace']

acc_matrix  = np.array([[rob[mk][dk]['accuracy'] for dk in dataset_keys]
                          for mk in model_keys])
rec_matrix  = np.array([[rob[mk][dk]['recall']   for dk in dataset_keys]
                          for mk in model_keys])

fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
axes = axes.flatten()
for ax, matrix, title, cmap in zip(
        axes,
        [acc_matrix, rec_matrix],
        ['Accuracy (%) — Model × Test Dataset',
         'Recall (%) — Model × Test Dataset'],
        ['YlGn', 'YlOrRd']):

    im  = ax.imshow(matrix, cmap=cmap, vmin=0.88, vmax=1.0, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(dataset_names, rotation=20, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels(model_names)
    ax.set_xlabel('Test Dataset (Mutation Type)')
    ax.set_ylabel('Trained Model')
    ax.set_title(title, fontweight='bold')

    for i in range(4):
        for j in range(4):
            val  = matrix[i, j]
            bold = 'bold' if i == j else 'normal'
            ring = '★' if i == j else ''
            ax.text(j, i, f'{val*100:.2f}%{ring}',
                    ha='center', va='center', fontsize=9.5,
                    fontweight=bold,
                    color='black' if val < 0.97 else 'white')

    plt.colorbar(im, ax=ax, format='%.0f%%',
                 ticks=np.linspace(0.88, 1.0, 7))

plt.suptitle('Robustness Evaluation — Adversarial Hardening vs Mutation Strategies',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_robustness_heatmap.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig5_robustness_heatmap.png', bbox_inches='tight')
plt.close()
print("   Saved fig5_robustness_heatmap.pdf/.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Grouped Bar Chart: Recall by Mutation Strategy
# ─────────────────────────────────────────────────────────────────────────────
print("[6/7] Grouped Recall Bar Chart...")

fig, ax = plt.subplots(figsize=(11, 6))
x      = np.arange(4)
width  = 0.2
mut_colors = [PALETTE['baseline'], PALETTE['random_insert'],
              PALETTE['noise_append'], PALETTE['random_replace']]
labels_bar = ['Baseline','+ Random Insert','+ Noise Append','+ Random Replace']

for idx, (mk, col, lbl) in enumerate(zip(model_keys, mut_colors, labels_bar)):
    recalls = [rob[mk][dk]['recall'] * 100 for dk in dataset_keys]
    bars    = ax.bar(x + idx*width, recalls, width, label=lbl, color=col, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, rotation=90)

ax.set_xlabel('Test Dataset (Mutation Type)')
ax.set_ylabel('Recall (%)')
ax.set_title('Malware Recall Across Models and Mutation Test Sets', fontweight='bold')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(dataset_names)
ax.set_ylim([85, 103])
ax.legend(loc='lower right', ncol=2)
ax.axhline(y=95, linestyle='--', color='red', alpha=0.4, linewidth=1,
           label='95% threshold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_recall_bar.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig6_recall_bar.png', bbox_inches='tight')
plt.close()
print("   Saved fig6_recall_bar.pdf/.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — ROC Curves: Mutation Models
# ─────────────────────────────────────────────────────────────────────────────
print("[7/7] ROC Curves — Mutation Models...")

fig, ax = plt.subplots(figsize=(7, 6))
mut_col_list = list(mut_colors)

for (name, ckpt), col in zip(MUTATION_CKPTS.items(), mut_col_list):
    probs, _, labels = get_probs_preds(ckpt, test_cnn, test_lstm, test_labels)
    fpr, tpr, _      = roc_curve(labels, probs)
    roc_auc          = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name}  (AUC = {roc_auc:.4f})', linewidth=2, color=col)

ax.plot([0,1],[0,1],'--', color='grey', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Adversarial Hardening Models')
ax.legend(loc='lower right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig7_roc_mutation.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig7_roc_mutation.png', bbox_inches='tight')
plt.close()
print("   Saved fig7_roc_mutation.pdf/.png")



# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Fusion Method Bar Chart: All 5 Metrics Side-by-Side
# ─────────────────────────────────────────────────────────────────────────────
print("[Bonus] Metric Comparison Bar Chart — Fusion Methods...")

metrics_to_plot = ['accuracy','precision','recall','f1_score','auc']
metric_labels2  = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
fusion_data     = {}
for name, jfile in EVAL_FILES.items():
    with open(jfile) as f: d = json.load(f)
    fusion_data[name] = [d['test_metrics'][m]*100 for m in metrics_to_plot]

x      = np.arange(len(metrics_to_plot))
width2 = 0.18
fig, ax = plt.subplots(figsize=(13, 6))
for idx, (name, vals) in enumerate(fusion_data.items()):
    bars = ax.bar(x + idx*width2, vals, width2, label=name,
                  color=list(PALETTE.values())[idx], alpha=0.87,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xticks(x + width2*1.5)
ax.set_xticklabels(metric_labels2)
ax.set_ylabel('Score (%)')
ax.set_ylim([88, 102])
ax.set_title('Fusion Method Comparison — All Metrics', fontweight='bold')
ax.legend(loc='lower right', ncol=2)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig8_bar_fusion_metrics.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig8_bar_fusion_metrics.png', bbox_inches='tight')
plt.close()
print("   Saved fig8_bar_fusion_metrics.pdf/.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"All figures saved to:\n  {FIG_DIR}")
print(f"{'='*60}")
print("\nFigure Summary:")
figs = sorted(FIG_DIR.glob('*.png'))
for f in figs:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:<45} {size_kb:>5} KB")
