# Dice Game Fraud Detection

A hardware-software project that simulates a dice game on an FPGA, transmits roll data to a PC via UART, and uses Machine Learning algorithms to detect fraudulent game modes.

---

## Project Overview

This project combines **digital hardware design** and **machine learning** to explore fraud detection in a controlled environment. A dice game is implemented in VHDL and runs on a **Basys 3 FPGA** board. Some game modes manipulate the dice rolls in subtle ways — the goal is to train two ML models to automatically detect when the game is being cheated.

The project covers the full pipeline:
- FPGA hardware design in VHDL
- Real-time serial communication (UART) between FPGA and PC
- Automated data collection
- Feature engineering
- Training and evaluating a **Random Forest** classifier
- Training and evaluating an **RNN/LSTM** classifier
- Head-to-head comparison of both models

---

## Game Modes

The game is controlled via 3 switches on the Basys 3 board. Each switch combination activates a different mode:

| Switch | Mode | Type | Description |
|--------|------|------|-------------|
| `000` | MANUAL | ✅ Fair | Roll on button press |
| `001` | AUTO | ✅ Fair | Automatic roll every 1 second |
| `010` | BIAS P1 | ❌ Fraud | P1 has 70% chance of rolling a 5 |
| `011` | CHAOS | ✅ Fair | Auto roll at 5x speed |
| `100` | ADAPTIVE BIAS | ❌ Fraud | P1 bias dynamically adjusts based on score history |

Every roll is transmitted to the PC over UART in the format:
```
P1:5,P2:2,M:2
```

---

## Project Structure

```
DiceGameFraudDetection/
│
├── fpga/
│   └── dice_game.vhd          # VHDL source — dice game + UART TX
│
├── ml/
│   ├── data_gatherer.py        # Serial logger — reads FPGA, saves to CSV
│   ├── feature_engineering.py  # Builds features for RF and sequences for RNN
│   ├── random_forest.py        # Random Forest classifier
│   ├── rnn_classifier.py       # RNN/LSTM classifier
│   └── comparison.py           # Head-to-head comparison and final report
│
├── data/                       # Auto-generated during data collection
│   ├── dice_rolls.csv
│   ├── dice_features.csv
│   └── dice_sequences.npy
│
└── results/                    # Auto-generated after training
    ├── rf_results/
    ├── rnn_results/
    └── comparison_results/
```

---

## Hardware Requirements

- **Basys 3 FPGA board** (Xilinx Artix-7)
- **Vivado Design Suite** (for synthesis and programming)
- USB cable (micro-USB for programming and UART communication)

### FPGA Design Details

- Random number generation via a **16-bit LFSR** (polynomial: x¹⁶ + x¹⁴ + x¹³ + x¹¹ + 1)
- Button debouncing with 10ms filter
- 4-digit 7-segment display showing P1 and P2 dice values
- **UART TX** at 115200 baud, 8N1, using the onboard FT2232 USB-UART bridge (pin D4)

---

## Software Requirements

- **Python 3.11** (recommended — PyTorch does not yet support 3.14+)
- Libraries: `pyserial`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`

### Installation

```powershell
py -3.11 -m pip install pyserial
py -3.11 -m pip install pandas
py -3.11 -m pip install numpy
py -3.11 -m pip install matplotlib
py -3.11 -m pip install seaborn
py -3.11 -m pip install scikit-learn
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## How to Run

> All scripts must be run from the **project root folder**, not from inside `ml/`.

### Step 1 — Program the FPGA

1. Open Vivado and set `dice_game.vhd` as the top module
2. Run Synthesis → Implementation → Generate Bitstream
3. Open Hardware Manager → Auto Connect → Program Device

### Step 2 — Collect Data

Set your COM port in `ml/data_gatherer.py`:
```python
COM_PORT = "COM3"  # adjust to your system
```

Then run:
```powershell
py -3.11 ml/data_gatherer.py
```

Collect data across all modes — aim for **3000+ total rolls**:

| Mode | Target rolls | Approx time |
|------|-------------|-------------|
| CHAOS `011` | 1000+ | ~4 min |
| AUTO `001` | 600+ | ~10 min |
| BIAS P1 `010` | 600+ | ~10 min |
| ADAPTIVE `100` | 600+ | ~10 min |
| MANUAL `000` | 200+ | Manual |

Press `Ctrl+C` to stop logging.

### Step 3 — Feature Engineering

```powershell
py -3.11 ml/feature_engineering.py
```

### Step 4 — Train Random Forest

```powershell
py -3.11 ml/random_forest.py
```

### Step 5 — Train RNN/LSTM

```powershell
py -3.11 ml/rnn_classifier.py
```

### Step 6 — Compare Models

```powershell
py -3.11 ml/comparison.py
```

---

## Machine Learning

### Feature Engineering

A **sliding window of 20 consecutive rolls** is used to compute statistical features for each sample:

| Feature | Description | Key for |
|---------|-------------|---------|
| `p1_5_freq` | How often P1 rolls exactly 5 | BIAS P1 mode |
| `p1_high_freq` | How often P1 rolls 4, 5, or 6 | ADAPTIVE mode |
| `p1_entropy` | Shannon entropy of P1 distribution | Both fraud modes |
| `p1_win_rate` | How often P1 beats P2 | Both fraud modes |
| `mean_diff` | Average (P1 - P2) | Both fraud modes |

### Random Forest

- 200 decision trees
- Balanced class weights to handle imbalance
- Outputs feature importance showing **which stats exposed the fraud**
- Best at detecting **static fraud** (BIAS P1 mode)

### RNN / LSTM

- 2-layer LSTM with 64 hidden units
- Input: raw sequences of `[p1, p2]` pairs normalized to `[0, 1]`
- No manual feature engineering — the model learns patterns itself
- Best at detecting **dynamic fraud** (ADAPTIVE mode) due to temporal memory

### Key Finding

Neither model is universally better — the best choice depends on the type of fraud:

```
BIAS P1 mode    → Random Forest wins  (static pattern, p1_5_freq is a clear giveaway)
ADAPTIVE mode   → RNN/LSTM wins       (shifting pattern over time, LSTM memory is the advantage)
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/rf_results/feature_importance.png` | Which features caught the fraud |
| `results/rf_results/roc_curve.png` | RF ROC curve |
| `results/rnn_results/training_history.png` | LSTM loss and AUC over epochs |
| `results/rnn_results/fraud_probability_dist.png` | How confidently the RNN separates fraud from fair |
| `results/comparison_results/metrics_comparison.png` | Side-by-side bar chart of all metrics |
| `results/comparison_results/roc_curves_combined.png` | Both ROC curves on one plot |
| `results/comparison_results/per_mode_auc.png` | AUC broken down per game mode |
| `results/comparison_results/final_report.txt` | Full text comparison report |

---

## UART Protocol

The FPGA transmits one line per roll over USB at **115200 baud, 8N1**:

```
P1:X,P2:Y,M:Z\n
```

Where:
- `X` = P1 dice value (1–6)
- `Y` = P2 dice value (1–6)
- `Z` = active mode (0–4)

No extra hardware is needed — the Basys 3 has an onboard USB-to-UART bridge (FT2232). The same USB cable used for programming carries the UART data on a separate virtual COM port.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
