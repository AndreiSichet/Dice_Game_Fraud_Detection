import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from torch.utils.data          import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (classification_report, confusion_matrix,roc_auc_score, roc_curve, ConfusionMatrixDisplay)


# CONFIGURATION

SEQUENCE_NPY = "../data/dice_sequences.npy"
RESULTS_DIR  = "../results/rnn_results"
RANDOM_SEED  = 42

# Training hyperparameters
SEQ_LEN      = 20
INPUT_SIZE   = 2      # [p1, p2] per roll
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 0.001

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# 1. DATASET CLASS

class DiceDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (N, SEQ_LEN, 2) — already normalized float in [0,1]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 2. MODEL DEFINITION

class FraudLSTM(nn.Module):
    """
    Two-layer LSTM followed by a fully connected classifier.

    Input  : (batch, seq_len=20, features=2)
    Output : (batch, 1)  — probability of fraud (sigmoid activated)

    Architecture:
      LSTM layer 1: 2   → 64 hidden units
      LSTM layer 2: 64  → 64 hidden units
      Dropout:      0.3 (regularization to prevent overfitting)
      FC layer:     64  → 1
      Sigmoid:      maps to [0, 1] probability
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(FraudLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True    # input shape: (batch, seq, features)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take only the LAST timestep — it has seen the entire sequence
        last_step = lstm_out[:, -1, :]     # shape: (batch, hidden_size)

        out = self.dropout(last_step)
        out = self.fc(out)                 # shape: (batch, 1)
        out = self.sigmoid(out)
        return out.squeeze(1)              # shape: (batch,)


# 3. LOAD DATA

def load_data():
    print(f"Loading sequences from {SEQUENCE_NPY}...")
    data = np.load(SEQUENCE_NPY, allow_pickle=True).item()
    X    = data['X']   # shape: (N, SEQ_LEN, 2)
    y    = data['y']   # shape: (N,)

    print(f"Total sequences : {len(y)}")
    print(f"Sequence shape  : {X.shape}")
    print(f"Fraud samples   : {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Fair samples    : {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    return X, y


# 4. BUILD DATALOADERS

def build_loaders(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\nTrain set : {len(y_train)} samples")
    print(f"Test set  : {len(y_test)} samples")

    train_dataset = DiceDataset(X_train, y_train)
    test_dataset  = DiceDataset(X_test,  y_test)

    # Weighted sampler to handle class imbalance during training
    class_counts  = np.bincount(y_train.astype(int))
    weights       = 1.0 / class_counts
    sample_weights = weights[y_train.astype(int)]
    sampler       = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False)

    return train_loader, test_loader, X_test, y_test


# 5. TRAINING LOOP

def train(model, train_loader, test_loader):
    criterion = nn.BCELoss()                          # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=5
    )

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_auc   = 0.0
    best_state = None

    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"{'Epoch':<7} {'Train Loss':>12} {'Val Loss':>10} {'Val AUC':>10}")
    print("-" * 45)

    for epoch in range(1, EPOCHS + 1):

        # --- Training phase ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation phase ---
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y_pred  = model(X_batch)
                val_losses.append(criterion(y_pred, y_batch).item())
                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_auc    = roc_auc_score(all_labels, all_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        # Save best model
        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<7} {train_loss:>12.4f} {val_loss:>10.4f} {val_auc:>10.4f}")

    print(f"\nBest validation AUC: {best_auc:.4f}")
    model.load_state_dict(best_state)  # restore best weights
    return model, history


# 6. EVALUATE

def evaluate(model, X_test, y_test):
    model.eval()
    dataset    = DiceDataset(X_test, y_test)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = model(X_batch.to(DEVICE)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc        = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*55)
    print("CLASSIFICATION REPORT (test set)")
    print("="*55)
    print(classification_report(all_labels, all_preds,
                                 target_names=['Fair', 'Fraud']))
    print(f"ROC-AUC Score : {auc:.4f}")

    return all_preds, all_probs, all_labels, auc


# 7. PLOTS

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax1.plot(epochs, history['train_loss'], label='Train Loss',
             color='steelblue', lw=2)
    ax1.plot(epochs, history['val_loss'],   label='Val Loss',
             color='crimson',   lw=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('BCE Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # AUC curve
    ax2.plot(epochs, history['val_auc'], color='seagreen', lw=2)
    ax2.axhline(max(history['val_auc']), color='gray',
                linestyle='--', alpha=0.7,
                label=f"Best AUC = {max(history['val_auc']):.4f}")
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('ROC-AUC', fontsize=11)
    ax2.set_title('Validation AUC over Training', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_confusion_matrix(y_true, y_pred):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Fair', 'Fraud'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Purples')
    ax.set_title('RNN/LSTM — Confusion Matrix', fontsize=13, fontweight='bold')
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

def plot_roc_curve(y_true, y_probs, auc):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='purple', lw=2,
            label=f'RNN/LSTM (AUC = {auc:.4f})')
    ax.plot([0,1], [0,1], color='gray', linestyle='--', label='Random baseline')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — RNN/LSTM', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_fraud_probability(y_probs, y_true):
    """Shows how confidently the model scores fraud vs fair rolls."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist([y_probs[y_true == 0], y_probs[y_true == 1]],
            bins=40, label=['Fair (actual)', 'Fraud (actual)'],
            color=['steelblue', 'crimson'], alpha=0.7, edgecolor='white')
    ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Decision threshold (0.5)')
    ax.set_xlabel('P(Fraud) predicted by RNN', fontsize=11)
    ax.set_ylabel('Number of samples', fontsize=11)
    ax.set_title('Fraud Probability Distribution\n(good model = two separated peaks)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fraud_probability_dist.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


# 8. SAVE REPORT

def save_report(y_true, y_pred, y_probs, auc, history):
    report_text = classification_report(y_true, y_pred,
                                        target_names=['Fair', 'Fraud'])
    lines = [
        "="*55,
        "RNN/LSTM — FRAUD DETECTION REPORT",
        "="*55,
        "",
        f"Architecture     : LSTM x{NUM_LAYERS} layers, {HIDDEN_SIZE} hidden units",
        f"Sequence length  : {SEQ_LEN} rolls",
        f"Dropout          : {DROPOUT}",
        f"Epochs trained   : {EPOCHS}",
        f"Best Val AUC     : {max(history['val_auc']):.4f}",
        f"Final Test AUC   : {auc:.4f}",
        "",
        "CLASSIFICATION REPORT:",
        report_text,
    ]
    path = os.path.join(RESULTS_DIR, "rnn_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved full report: {path}")


# MAIN

def main():
    print("="*55)
    print("  Dice Fraud Detection — RNN/LSTM Classifier")
    print("="*55)

    X, y                                      = load_data()
    train_loader, test_loader, X_test, y_test = build_loaders(X, y)

    model = FraudLSTM().to(DEVICE)
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    model, history = train(model, train_loader, test_loader)

    y_pred, y_probs, y_true, auc = evaluate(model, X_test, y_test)

    print("\nGenerating plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_probs, auc)
    plot_fraud_probability(y_probs, y_true)
    save_report(y_true, y_pred, y_probs, auc, history)

    # Save model weights for the comparison step
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, "best_model.pt"))
    print(f"\nModel weights saved to: {RESULTS_DIR}/best_model.pt")

    print("\n" + "="*55)
    print("All done. Results saved to:", os.path.abspath(RESULTS_DIR))
    print("="*55)

if __name__ == "__main__":
    main()