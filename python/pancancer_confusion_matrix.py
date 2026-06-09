"""Pan-cancer 8x8 confusion matrix figure from existing CSV."""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(base_dir), "outputs")

df = pd.read_csv(os.path.join(OUTPUT_DIR, "MultiClass_PanCancer_ConfMatrix.csv"), index_col=0)
labels = [c.replace("pred_TCGA-", "") for c in df.columns]
cm = df.values.astype(int)

# Normalise by row (true class) for colour, show raw counts as text
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=12)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel('Predicted label', fontsize=13)
ax.set_ylabel('True label', fontsize=13)
ax.set_title('Pan-Cancer Multi-Class RF — OOF Confusion Matrix (8 tumour types)\n'
             'Colour = row-normalised rate  |  Number = raw OOF count', fontsize=13, pad=14)

for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm[i, j]
        color = 'white' if cm_norm[i, j] > 0.55 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', fontsize=11,
                color=color, fontweight='bold' if i == j else 'normal')

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('Recall (row-normalised)', fontsize=11)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "figures", "confusion_matrix_pancancer.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
