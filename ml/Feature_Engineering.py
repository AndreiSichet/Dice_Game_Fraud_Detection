import pandas as pd
import numpy as np
from collections import Counter


# CONFIGURATION

INPUT_CSV    = "../data/dice_rolls.csv"
OUTPUT_CSV   = "../data/dice_features.csv"   # RF will train on this
SEQUENCE_NPY = "../data/dice_sequences.npy"  # RNN will train on this
SEQ_LEN      = 20  # how many consecutive rolls to look back


# 1. LOAD RAW DATA

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rolls from {filepath}")
    print(f"\nRoll distribution by mode:")
    for mode, count in sorted(Counter(df['mode']).items()):
        label = 'FRAUD' if df[df['mode']==mode]['label'].iloc[0] == 1 else 'fair '
        print(f"  Mode {mode} [{label}]: {count} rolls")
    print(f"\nFraud rolls : {df['label'].sum()}")
    print(f"Fair rolls  : {(df['label']==0).sum()}")
    return df


# 2. FEATURE ENGINEERING (sliding window)
#
# For each roll at position i, we look at the last SEQ_LEN rolls
# and compute statistical features from that window.
# This is what the Random Forest will learn from.
#
# Features computed per window:
#   p1_mean       — average P1 roll (high in bias mode)
#   p1_std        — how consistent P1 is (low when always rolling 5)
#   p2_mean       — average P2 roll (should be ~3.5 always)
#   p2_std        — P2 spread
#   p1_win_rate   — how often P1 > P2 (high in fraud modes)
#   p1_5_freq     — how often P1 rolls exactly 5 (key for BIAS mode)
#   p1_high_freq  — how often P1 rolls 4,5,6 (key for ADAPTIVE mode)
#   mean_diff     — average (p1 - p2), positive = P1 favored
#   p1_4_freq     — frequency of P1 rolling 4
#   p1_6_freq     — frequency of P1 rolling 6
#   p1_low_freq   — how often P1 rolls 1,2,3 (should be ~50% if fair)
#   p2_high_freq  — how often P2 rolls 4,5,6 (sanity check, should be ~50%)
#   roll_entropy  — entropy of P1 distribution (low entropy = biased)

def compute_entropy(series):
    """Shannon entropy of a roll series. Lower = more biased."""
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts + 1e-9))

def engineer_features(df, window=SEQ_LEN):
    print(f"\nEngineering features with window size = {window}...")
    rows = []

    for i in range(window, len(df)):
        w = df.iloc[i - window : i]  # the window of 'window' rolls before roll i
        p1 = w['p1']
        p2 = w['p2']

        rows.append({
            # P1 statistics
            'p1_mean'      : p1.mean(),
            'p1_std'       : p1.std(),
            'p1_5_freq'    : (p1 == 5).mean(),          # BIAS mode signature
            'p1_4_freq'    : (p1 == 4).mean(),
            'p1_6_freq'    : (p1 == 6).mean(),
            'p1_high_freq' : (p1 >= 4).mean(),          # ADAPTIVE mode signature
            'p1_low_freq'  : (p1 <= 3).mean(),
            'p1_entropy'   : compute_entropy(p1),       # low = biased

            # P2 statistics (should be ~fair in all modes)
            'p2_mean'      : p2.mean(),
            'p2_std'       : p2.std(),
            'p2_high_freq' : (p2 >= 4).mean(),

            # Comparative statistics
            'p1_win_rate'  : (p1 > p2).mean(),          # P1 wins more in fraud modes
            'mean_diff'    : (p1 - p2).mean(),           # positive = P1 favored
            'diff_std'     : (p1 - p2).std(),

            # Label (majority vote in window — uses last roll's label)
            'label'        : df.iloc[i]['label'],
            'mode'         : df.iloc[i]['mode']
        })

    features = pd.DataFrame(rows)
    print(f"Generated {len(features)} feature rows ({len(df) - window} skipped for warm-up)")
    return features


# 3. SEQUENCE BUILDER (for RNN)
#
# Instead of engineered features, the RNN sees the raw sequence directly.
# Output: numpy array of shape (N, SEQ_LEN, 2)
#   N        = number of samples
#   SEQ_LEN  = 20 consecutive rolls per sample
#   2        = [p1, p2] normalized to 0-1 range

def build_sequences(df, window=SEQ_LEN):
    print(f"\nBuilding RNN sequences with window size = {window}...")
    X, y = [], []

    # Normalize dice values from [1,6] to [0,1]
    p1_norm = (df['p1'].values - 1) / 5.0
    p2_norm = (df['p2'].values - 1) / 5.0
    labels  = df['label'].values

    for i in range(window, len(df)):
        sequence = np.column_stack([
            p1_norm[i - window : i],
            p2_norm[i - window : i]
        ])  # shape: (SEQ_LEN, 2)
        X.append(sequence)
        y.append(labels[i])

    X = np.array(X)  # shape: (N, SEQ_LEN, 2)
    y = np.array(y)  # shape: (N,)

    print(f"Sequence array shape : {X.shape}")
    print(f"Label array shape    : {y.shape}")
    return X, y


# 4. DATASET SUMMARY

def print_summary(features):
    print("\n" + "="*55)
    print("FEATURE DATASET SUMMARY")
    print("="*55)
    print(f"Total samples    : {len(features)}")
    print(f"Fraud samples    : {features['label'].sum()}")
    print(f"Fair samples     : {(features['label']==0).sum()}")
    balance = features['label'].mean() * 100
    print(f"Fraud ratio      : {balance:.1f}%")

    if balance < 30 or balance > 70:
        print("\n  [WARN] Dataset is imbalanced (ideally 40-60% fraud).")
        print("         Collect more rolls from the underrepresented modes.")
    else:
        print("\n  [OK] Dataset balance looks good for training.")

    print("\nFeature means by label:")
    cols = ['p1_mean', 'p1_5_freq', 'p1_high_freq', 'p1_win_rate', 'p1_entropy']
    print(features.groupby('label')[cols].mean().round(3).to_string())


# MAIN

def main():
    print("="*55)
    print("  Dice Fraud — Feature Engineering Pipeline")
    print("="*55)

    # Load
    df = load_data(INPUT_CSV)

    if len(df) < SEQ_LEN + 10:
        print(f"\n[ERROR] Need at least {SEQ_LEN + 10} rolls, only have {len(df)}.")
        print("        Run the serial logger to collect more data first.")
        return

    # Build features for Random Forest
    features = engineer_features(df)
    features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved RF features to: {OUTPUT_CSV}")

    # Build sequences for RNN
    X, y = build_sequences(df)
    np.save(SEQUENCE_NPY, {'X': X, 'y': y})
    print(f"Saved RNN sequences to: {SEQUENCE_NPY}")

    # Summary
    print_summary(features)

    print("\nDone. You now have:")
    print(f"  {OUTPUT_CSV}      → input for Random Forest")
    print(f"  {SEQUENCE_NPY}    → input for RNN/LSTM")

if __name__ == "__main__":
    main()