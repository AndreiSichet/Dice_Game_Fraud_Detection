import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics           import (classification_report, confusion_matrix,roc_auc_score, roc_curve, ConfusionMatrixDisplay)                                   
from sklearn.preprocessing     import label_binarize


# CONFIGURATION

INPUT_CSV    = "../data/dice_features.csv"
RESULTS_DIR  = "../results/rf_results"        # all plots and reports saved here
RANDOM_SEED  = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# 1. LOAD DATA

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")

    X = df.drop(columns=['label','mode'])
    y = df['label']

    print(f"\nClass distribution:")
    print(f"  Fair  (0): {(y==0).sum()} samples ({(y==0).mean()*100:.1f}%)")
    print(f"  Fraud (1): {(y==1).sum()} samples ({(y==1).mean()*100:.1f}%)")
    return X, y

# 2. TRAIN / TEST SPLIT

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,       # 80% train, 20% test
        random_state = RANDOM_SEED,
        stratify     = y          # keep class ratio in both splits
    )
    print(f"\nTrain set : {len(X_train)} samples")
    print(f"Test set  : {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


# 3. TRAIN RANDOM FOREST

def train_model(X_train, y_train):
    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators = 200,    # number of trees — more = more stable, slower
        max_depth    = None,   # let trees grow fully
        min_samples_split = 5,
        min_samples_leaf  = 2,
        class_weight = 'balanced',  # handles class imbalance automatically
        random_state = RANDOM_SEED,
        n_jobs       = -1      # use all CPU cores
    )

    rf.fit(X_train, y_train)
    print("Training complete.")
    return rf


# 4. EVALUATE

def evaluate(rf, X_train, X_test, y_train, y_test):
    y_pred      = rf.predict(X_test)
    y_pred_prob = rf.predict_proba(X_test)[:, 1]  # probability of fraud

    print("\n" + "="*55)
    print("CLASSIFICATION REPORT (test set)")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=['Fair', 'Fraud']))

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score : {auc:.4f}")

    # 5-fold cross validation on full dataset for more robust estimate
    print("\nRunning 5-fold cross-validation...")
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    cv_scores = cross_val_score(rf, X_all, y_all, cv=5,
                                scoring='roc_auc', n_jobs=-1)
    print(f"Cross-val AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return y_pred, y_pred_prob, auc


# 5. PLOTS

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Fair', 'Fraud'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Random Forest — Confusion Matrix', fontsize=13, fontweight='bold')

    # Annotate with rates
    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(
        f"Predicted\n\nTrue Negatives: {tn}  |  False Positives: {fp}"
        f"\nFalse Negatives: {fn}  |  True Positives: {tp}",
        fontsize=9
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_roc_curve(y_test, y_pred_prob, auc):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'Random Forest (AUC = {auc:.4f})')
    ax.plot([0,1], [0,1], color='gray', linestyle='--', label='Random baseline')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — Random Forest', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_feature_importance(rf, feature_names):
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]  # sort descending

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['crimson' if importances[i] > np.median(importances)
              else 'steelblue' for i in indices]

    ax.bar(range(len(indices)),
           importances[indices],
           color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Importance Score', fontsize=11)
    ax.set_title('Feature Importance — Which stats exposed the fraud?',
                 fontsize=13, fontweight='bold')
    ax.axhline(np.median(importances), color='gray',
               linestyle='--', alpha=0.7, label='Median importance')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

    # Print ranked list
    print("\nFeature importance ranking:")
    print(f"  {'Rank':<5} {'Feature':<20} {'Importance':>10}")
    print("  " + "-"*38)
    for rank, i in enumerate(indices, 1):
        bar = "#" * int(importances[i] * 200)
        print(f"  {rank:<5} {feature_names[i]:<20} {importances[i]:>10.4f}  {bar}")

def plot_decision_boundary_2d(rf, X_test, y_test):
    """
    Visualizes the decision boundary using the two most
    important features so you can see how the RF splits the space.
    """
    importances  = rf.feature_importances_
    top2_idx     = np.argsort(importances)[::-1][:2]
    feat_names   = X_test.columns.tolist()
    f1, f2       = feat_names[top2_idx[0]], feat_names[top2_idx[1]]

    X2 = X_test[[f1, f2]].values
    y2 = y_test.values

    # Grid
    x_min, x_max = X2[:,0].min()-0.1, X2[:,0].max()+0.1
    y_min, y_max = X2[:,1].min()-0.1, X2[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Fill remaining features with their mean for the grid prediction
    grid_full = np.tile(X_test.mean().values, (xx.ravel().shape[0], 1))
    grid_full[:, top2_idx[0]] = xx.ravel()
    grid_full[:, top2_idx[1]] = yy.ravel()

    Z = rf.predict_proba(grid_full)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(xx, yy, Z, levels=50, cmap='RdYlGn_r', alpha=0.7)
    plt.colorbar(cp, ax=ax, label='P(Fraud)')
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)

    scatter = ax.scatter(X2[:,0], X2[:,1], c=y2,
                         cmap='bwr', edgecolors='k',
                         linewidths=0.4, s=30, alpha=0.8)
    ax.set_xlabel(f1, fontsize=11)
    ax.set_ylabel(f2, fontsize=11)
    ax.set_title(f'Decision Boundary (top 2 features)\nBlack line = 50% fraud threshold',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "decision_boundary.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


# 6. SAVE SUMMARY REPORT

def save_report(rf, y_test, y_pred, y_pred_prob, auc, X):
    report_text = classification_report(y_test, y_pred,
                                        target_names=['Fair', 'Fraud'])
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    feat_names  = X.columns.tolist()

    lines = [
        "="*55,
        "RANDOM FOREST — FRAUD DETECTION REPORT",
        "="*55,
        "",
        f"Training samples : {len(y_test)*4}  (approx)",
        f"Test samples     : {len(y_test)}",
        f"ROC-AUC Score    : {auc:.4f}",
        "",
        "CLASSIFICATION REPORT:",
        report_text,
        "",
        "FEATURE IMPORTANCE (ranked):",
    ]
    for rank, i in enumerate(indices, 1):
        lines.append(f"  {rank}. {feat_names[i]:<20} {importances[i]:.4f}")

    report = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "rf_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nSaved full report: {path}")


# MAIN

def main():
    print("="*55)
    print("  Dice Fraud Detection — Random Forest")
    print("="*55)

    X, y                             = load_data(INPUT_CSV)
    X_train, X_test, y_train, y_test = split_data(X, y)
    rf                               = train_model(X_train, y_train)
    y_pred, y_pred_prob, auc         = evaluate(rf, X_train, X_test,
                                                 y_train, y_test)

    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob, auc)
    plot_feature_importance(rf, X.columns.tolist())
    plot_decision_boundary_2d(rf, X_test, y_test)
    save_report(rf, y_test, y_pred, y_pred_prob, auc, X)

    print("\n" + "="*55)
    print("All done. Results saved to:", os.path.abspath(RESULTS_DIR))
    print("="*55)

if __name__ == "__main__":
    main()