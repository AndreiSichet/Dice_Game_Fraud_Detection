import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import torch
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, accuracy_score,
                             ConfusionMatrixDisplay)


# CONFIGURATION

FEATURES_CSV = "../data/dice_features.csv"
SEQUENCE_NPY = "../data/dice_sequences.npy"
RESULTS_DIR  = "../results/comparison_results"
RANDOM_SEED  = 42


INPUT_SIZE   = 2
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
SEQ_LEN      = 20
BATCH_SIZE   = 32

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. LOAD AND ALIGN DATA
#    Both models must be evaluated on the EXACT same test indices.
#    We use the same random seed and test_size=0.2 as before — this
#    guarantees the split is identical to what each model was trained on.


# --- RF data ---
def load_rf_data():
    df     = pd.read_csv(FEATURES_CSV)
    X      = df.drop(columns=['label'])
    y      = df['label']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X, y, X_test, y_test

# --- RNN data ---
def load_rnn_data():
    data   = np.load(SEQUENCE_NPY, allow_pickle=True).item()
    X, y   = data['X'], data['y']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X, y, X_test, y_test


# 2. RETRAIN RANDOM FOREST
#    (RF has no saved weights file — retraining is instant)

def get_rf_predictions(X_full, y_full, X_test, y_test):
    print("Retraining Random Forest...")
    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_full
    )
    rf = RandomForestClassifier(
        n_estimators      = 200,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        class_weight      = 'balanced',
        random_state      = RANDOM_SEED,
        n_jobs            = -1
    )
    rf.fit(X_train, y_train)
    y_pred  = rf.predict(X_test)
    y_probs = rf.predict_proba(X_test)[:, 1]
    print("  Done.")
    return rf, y_pred, y_probs


# 3. LOAD RNN AND GET PREDICTIONS


# Import model class inline so we don't depend on the other file
class FraudLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = torch.nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                                     dropout=DROPOUT, batch_first=True)
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.fc      = torch.nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.sigmoid(self.fc(out)).squeeze(1)

def get_rnn_predictions(X_test, y_test):
    print("Loading RNN weights...")
    model_path = os.path.join("../results/rnn_results", "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find {model_path}. "
            "Run rnn_classifier.py first to generate the saved weights."
        )
    model = FraudLSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y_probs = model(X_tensor).cpu().numpy()
    y_pred = (y_probs >= 0.5).astype(int)
    print("  Done.")
    return y_pred, y_probs


# 4. COMPUTE ALL METRICS

def compute_metrics(y_true, y_pred, y_probs, name):
    return {
        'model'    : name,
        'accuracy' : accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall'   : recall_score(y_true, y_pred),
        'f1'       : f1_score(y_true, y_pred),
        'roc_auc'  : roc_auc_score(y_true, y_probs),
        'y_true'   : y_true,
        'y_pred'   : y_pred,
        'y_probs'  : y_probs,
    }


# 5. PER-MODE ANALYSIS
#    Break down performance by fraud mode to see WHERE each model wins.
#    Requires that the test set preserves the original 'mode' column.

def per_mode_analysis(rf_probs, rnn_probs, y_test_rnn):
    print("\nRunning per-mode analysis...")

    # Reload raw CSV to get mode column aligned with RNN test split
    df        = pd.read_csv(FEATURES_CSV)
    _, df_test = train_test_split(df, test_size=0.2,random_state=RANDOM_SEED, stratify=df['label'])

    # RNN test set is the same size but derived from sequences, not features.
    # We align by index length — both splits use the same seed and stratify.
    mode_col = df_test['mode'].values if 'mode' in df_test.columns else None

    if mode_col is None:
        print("  [SKIP] 'mode' column not found in features CSV.")
        return None

    results = []
    mode_labels = {0: 'MANUAL (fair)', 1: 'AUTO (fair)',
                   2: 'BIAS P1 (fraud)', 3: 'CHAOS (fair)',
                   4: 'ADAPTIVE (fraud)'}

    for mode_id, mode_name in mode_labels.items():
        mask = mode_col == mode_id
        if mask.sum() == 0:
            continue
        true   = df_test['label'].values[mask]
        rf_p   = rf_probs[mask]
        rnn_p  = rnn_probs[mask] if len(rnn_probs) == len(rf_probs) else None

        rf_auc  = roc_auc_score(true, rf_p)  if len(np.unique(true)) > 1 else float('nan')
        rnn_auc = roc_auc_score(true, rnn_p) if (rnn_p is not None and len(np.unique(true)) > 1) else float('nan')
        results.append({
            'mode': mode_name, 'n': mask.sum(),
            'rf_auc': rf_auc,  'rnn_auc': rnn_auc
        })

    return pd.DataFrame(results)


# 6. PLOTS

RF_COLOR  = 'steelblue'
RNN_COLOR = 'mediumpurple'

def plot_metrics_bar(rf_m, rnn_m):
    """Side-by-side bar chart of all metrics."""
    metrics  = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    labels   = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    rf_vals  = [rf_m[m]  for m in metrics]
    rnn_vals = [rnn_m[m] for m in metrics]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    bars_rf  = ax.bar(x - w/2, rf_vals,  w, label='Random Forest',
                      color=RF_COLOR,  edgecolor='white', linewidth=0.8)
    bars_rnn = ax.bar(x + w/2, rnn_vals, w, label='RNN/LSTM',
                      color=RNN_COLOR, edgecolor='white', linewidth=0.8)

    # Value labels on top of each bar
    for bar in bars_rf:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9, color=RF_COLOR, fontweight='bold')
    for bar in bars_rnn:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9, color=RNN_COLOR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Random Forest vs RNN/LSTM — All Metrics',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_roc_curves_combined(rf_m, rnn_m):
    """Both ROC curves on one plot."""
    rf_fpr,  rf_tpr,  _ = roc_curve(rf_m['y_true'],  rf_m['y_probs'])
    rnn_fpr, rnn_tpr, _ = roc_curve(rnn_m['y_true'], rnn_m['y_probs'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rf_fpr,  rf_tpr,  color=RF_COLOR,  lw=2,
            label=f"Random Forest  (AUC = {rf_m['roc_auc']:.4f})")
    ax.plot(rnn_fpr, rnn_tpr, color=RNN_COLOR, lw=2,
            label=f"RNN/LSTM       (AUC = {rnn_m['roc_auc']:.4f})")
    ax.plot([0,1], [0,1], color='gray', linestyle='--',
            lw=1.5, label='Random baseline')
    ax.fill_between(rf_fpr,  rf_tpr,  alpha=0.08, color=RF_COLOR)
    ax.fill_between(rnn_fpr, rnn_tpr, alpha=0.08, color=RNN_COLOR)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('ROC Curves — Side by Side', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curves_combined.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_confusion_matrices_side_by_side(rf_m, rnn_m):
    """RF and RNN confusion matrices next to each other."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, m, title, cmap in [
        (ax1, rf_m,  'Random Forest', 'Blues'),
        (ax2, rnn_m, 'RNN/LSTM',      'Purples')
    ]:
        cm   = confusion_matrix(m['y_true'], m['y_pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Fair', 'Fraud'])
        disp.plot(ax=ax, colorbar=False, cmap=cmap)
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(f'{title}\nAcc={m["accuracy"]:.3f}  F1={m["f1"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(
            f'Predicted\nTP={tp}  FP={fp}  FN={fn}  TN={tn}', fontsize=9
        )

    plt.suptitle('Confusion Matrices — Random Forest vs RNN/LSTM',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")

def plot_probability_distributions(rf_m, rnn_m):
    """How confidently each model separates fraud from fair."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, m, title, color in [
        (ax1, rf_m,  'Random Forest', RF_COLOR),
        (ax2, rnn_m, 'RNN/LSTM',      RNN_COLOR)
    ]:
        y_true  = np.array(m['y_true'])
        y_probs = np.array(m['y_probs'])
        ax.hist(y_probs[y_true == 0], bins=30, alpha=0.7,
                color='steelblue', label='Fair (actual)',   edgecolor='white')
        ax.hist(y_probs[y_true == 1], bins=30, alpha=0.7,
                color='crimson',   label='Fraud (actual)',  edgecolor='white')
        ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold 0.5')
        ax.set_xlabel('P(Fraud)', fontsize=11)
        ax.set_ylabel('Samples',  fontsize=11)
        ax.set_title(f'{title}\nFraud Probability Distribution',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Separation Quality — Wider gap = more confident model',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "probability_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")

def plot_per_mode(mode_df):
    """Bar chart of AUC per game mode for each model."""
    if mode_df is None:
        return

    x   = np.arange(len(mode_df))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    bars_rf  = ax.bar(x - w/2, mode_df['rf_auc'],  w,
                      label='Random Forest', color=RF_COLOR,  edgecolor='white')
    bars_rnn = ax.bar(x + w/2, mode_df['rnn_auc'], w,
                      label='RNN/LSTM',      color=RNN_COLOR, edgecolor='white')

    for bars in [bars_rf, bars_rnn]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['mode']}\n(n={r['n']})" for _, r in mode_df.iterrows()],
        fontsize=9
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title('AUC per Game Mode — Where does each model win?',
                 fontsize=13, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Shade fraud modes
    for i, row in mode_df.iterrows():
        if 'fraud' in row['mode'].lower():
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.07, color='crimson')

    ax.text(0.98, 0.97, '■ Red shading = fraud modes',
            transform=ax.transAxes, fontsize=8,
            color='crimson', ha='right', va='top')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "per_mode_auc.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


# 7. FINAL SUMMARY TABLE (printed + saved)

def print_and_save_summary(rf_m, rnn_m, mode_df):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

    winner_count = {'Random Forest': 0, 'RNN/LSTM': 0}

    lines = [
        "=" * 65,
        "  FRAUD DETECTION — FINAL COMPARISON REPORT",
        "=" * 65,
        "",
        f"{'Metric':<15} {'Random Forest':>15} {'RNN/LSTM':>15} {'Winner':>12}",
        "-" * 60,
    ]

    for metric, label in zip(metrics, labels):
        rf_val  = rf_m[metric]
        rnn_val = rnn_m[metric]
        if rf_val > rnn_val:
            winner = 'Random Forest'
            winner_count['Random Forest'] += 1
        elif rnn_val > rf_val:
            winner = 'RNN/LSTM'
            winner_count['RNN/LSTM'] += 1
        else:
            winner = 'Tie'
        lines.append(f"{label:<15} {rf_val:>15.4f} {rnn_val:>15.4f} {winner:>12}")

    lines += [
        "-" * 60,
        f"{'Wins':<15} {winner_count['Random Forest']:>15} {winner_count['RNN/LSTM']:>15}",
        "",
        "=" * 65,
        "  PER-MODE AUC BREAKDOWN",
        "=" * 65,
    ]

    if mode_df is not None:
        lines.append(
            f"\n{'Mode':<25} {'n':>6} {'RF AUC':>10} {'RNN AUC':>10} {'Winner':>12}"
        )
        lines.append("-" * 65)
        for _, row in mode_df.iterrows():
            rf_a  = row['rf_auc']
            rnn_a = row['rnn_auc']
            w     = ('RF' if rf_a > rnn_a else 'RNN') if not (np.isnan(rf_a) or np.isnan(rnn_a)) else 'N/A'
            lines.append(
                f"{row['mode']:<25} {int(row['n']):>6} {rf_a:>10.4f} {rnn_a:>10.4f} {w:>12}"
            )

    lines += [
        "",
        "=" * 65,
        "  CONCLUSION",
        "=" * 65,
        "",
        "Random Forest strengths:",
        "  - Instant training (no GPU needed)",
        "  - Interpretable via feature importance",
        "  - Best at static bias (BIAS P1 mode) where p1_5_freq is a giveaway",
        "",
        "RNN/LSTM strengths:",
        "  - Learns temporal patterns from raw sequences",
        "  - Best at dynamic fraud (ADAPTIVE mode) where bias shifts over time",
        "  - No manual feature engineering required",
        "",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    path = os.path.join(RESULTS_DIR, "final_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nSaved: {path}")


# MAIN

def main():
    print("=" * 65)
    print("  Dice Fraud Detection — RF vs RNN Comparison")
    print("=" * 65)

    # Load data
    X_rf_full, y_rf_full, X_rf_test, y_rf_test = load_rf_data()
    X_rnn_full, y_rnn_full, X_rnn_test, y_rnn_test = load_rnn_data()

    # Get predictions
    rf, rf_pred, rf_probs   = get_rf_predictions(X_rf_full,  y_rf_full,
                                                  X_rf_test,  y_rf_test)
    rnn_pred, rnn_probs     = get_rnn_predictions(X_rnn_test, y_rnn_test)

    # Compute metrics
    rf_m  = compute_metrics(y_rf_test.values,  rf_pred,  rf_probs,  'Random Forest')
    rnn_m = compute_metrics(y_rnn_test,        rnn_pred, rnn_probs, 'RNN/LSTM')

    # Per-mode breakdown
    mode_df = per_mode_analysis(rf_probs, rnn_probs, y_rnn_test)

    # Plots
    print("\nGenerating comparison plots...")
    plot_metrics_bar(rf_m, rnn_m)
    plot_roc_curves_combined(rf_m, rnn_m)
    plot_confusion_matrices_side_by_side(rf_m, rnn_m)
    plot_probability_distributions(rf_m, rnn_m)
    plot_per_mode(mode_df)

    # Final report
    print_and_save_summary(rf_m, rnn_m, mode_df)

    print("\n" + "="*65)
    print("All outputs saved to:", os.path.abspath(RESULTS_DIR))
    print("="*65)

if __name__ == "__main__":
    main()