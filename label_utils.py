# label_utils.py
import os
import pandas as pd

LABEL_MAP = {
    "A": 1,   # Alzheimer
    "C": 0    # Control
}

def load_file_label_pairs(root_dir, participants_tsv):
    df = pd.read_csv(participants_tsv, sep="\t")

    files, labels = [], []

    for _, row in df.iterrows():
        pid = row["participant_id"]   # e.g. sub-046
        group = row["Group"]

        if group not in LABEL_MAP:
            continue

        eeg_dir = os.path.join(root_dir, pid, "eeg")
        if not os.path.isdir(eeg_dir):
            raise FileNotFoundError(f"Missing eeg folder for {pid}")

        eeg_files = [
            f for f in os.listdir(eeg_dir)
            if f.endswith("_eeg.set")
        ]

        if len(eeg_files) != 1:
            raise ValueError(f"{pid}: expected 1 .set file, found {eeg_files}")

        files.append(os.path.join(eeg_dir, eeg_files[0]))
        labels.append(LABEL_MAP[group])

    return files, labels
