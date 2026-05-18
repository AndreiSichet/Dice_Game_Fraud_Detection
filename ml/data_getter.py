import serial
import serial.tools.list_ports
import csv
import os
from datetime import datetime


# CONFIGURATION — change COM_PORT to match your system

COM_PORT   = "COM4"     
                        
                        
BAUD_RATE  = 115200
OUTPUT_CSV = "../data/dice_rolls.csv"   

# Fraud label map — matches your VHDL switch modes
FRAUD_MODES = {0: 0, 1: 0, 2: 1, 3: 0, 4: 1}


# HELPER: find available COM ports and print them (useful for first run)

def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  No COM ports found.")
    for p in ports:
        print(f"  {p.device} — {p.description}")


# HELPER: parse one line from FPGA
# Input:  "P1:5,P2:2,M:2"
# Output: (5, 2, 2) or None if malformed

def parse_line(line):
    try:
        # Strip whitespace and split on comma
        parts = line.strip().split(",")
        if len(parts) != 3:
            return None

        p1   = int(parts[0].split(":")[1])  # "P1:5" → 5
        p2   = int(parts[1].split(":")[1])  # "P2:2" → 2
        mode = int(parts[2].split(":")[1])  # "M:2"  → 2

        # Basic sanity check
        if not (1 <= p1 <= 6 and 1 <= p2 <= 6 and 0 <= mode <= 4):
            return None

        return p1, p2, mode

    except (ValueError, IndexError):
        return None


# HELPER: write CSV header only if file is new/empty

def init_csv(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
    f = open(filepath, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["timestamp", "p1", "p2", "mode", "label"])
        f.flush()
        print(f"  Created new file: {filepath}")
    else:
        print(f"  Appending to existing file: {filepath}")
    return f, writer


# MAIN

def main():
    print("=" * 55)
    print("  FPGA Dice Game — Serial Logger")
    print("=" * 55)

    # Show available ports to help user pick the right one
    print("\nAvailable COM ports:")
    list_ports()
    print(f"\nConnecting to {COM_PORT} at {BAUD_RATE} baud...")

    # Open CSV
    csv_file, writer = init_csv(OUTPUT_CSV)
    roll_count = 0

    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=2) as ser:
            print(f"Connected. Listening for rolls. Press Ctrl+C to stop.\n")
            print(f"{'#':<6} {'Timestamp':<22} {'P1':>4} {'P2':>4} {'Mode':>6} {'Label':>6}")
            print("-" * 55)

            while True:
                raw = ser.readline()

                # Skip empty reads (timeout)
                if not raw:
                    continue

                # Decode bytes to string
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue  # skip corrupted bytes at startup

                # Skip empty lines
                if not line:
                    continue

                # Parse
                result = parse_line(line)
                if result is None:
                    print(f"  [WARN] Could not parse: '{line}'")
                    continue

                p1, p2, mode = result
                label = FRAUD_MODES.get(mode, -1)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Write to CSV
                writer.writerow([timestamp, p1, p2, mode, label])
                csv_file.flush()  # write immediately, don't buffer

                # Print to terminal
                roll_count += 1
                fraud_tag = " ← FRAUD" if label == 1 else ""
                print(f"{roll_count:<6} {timestamp:<22} {p1:>4} {p2:>4} {mode:>6} {label:>6}{fraud_tag}")

    except serial.SerialException as e:
        print(f"\n[ERROR] Serial port error: {e}")
        print("  Check that the correct COM port is set and the board is connected.")

    except KeyboardInterrupt:
        print(f"\n\nStopped by user.")
        print(f"Total rolls recorded: {roll_count}")
        print(f"Saved to: {os.path.abspath(OUTPUT_CSV)}")

    finally:
        csv_file.close()

if __name__ == "__main__":
    main()