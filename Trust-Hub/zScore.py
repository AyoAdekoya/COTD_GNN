# normalizer

import glob
import csv
import numpy as np

# Global variable to store the computed stats
scoap_stats = {
    "cc0": {"mean": None, "std": None},
    "cc1": {"mean": None, "std": None},
    "co": {"mean": None, "std": None},
}

def compute_scoap_stats(folder_path="Scoap_Scores"):
    """
    Computes and stores mean and std of CC0, CC1, CO across all csv files in given folder.
    """
    cc0_vals, cc1_vals, co_vals = [], [], []

    file_list = glob.glob(f"{folder_path}/gate_scores*.csv")
    if not file_list:
        print("No files found. Make sure relative path is correct")

    for file_path in file_list:
        with open(file_path, "r", newline="") as f:
            rdr = csv.reader(f)
            next(rdr)  # skip header
            for row in rdr:
                if not row:
                    continue
                try:
                    _, cc0, cc1, co, *_ = row
                    # Convert to float, handle "#INF"
                    cc0_f = float(10000) if cc0 == "#INF" else float(cc0)
                    cc1_f = float(10000) if cc1 == "#INF" else float(cc1)
                    co_f  = float(10000) if co == "#INF" else float(co)

                    if not np.isinf(cc0_f):
                        cc0_vals.append(cc0_f)
                    if not np.isinf(cc1_f):
                        cc1_vals.append(cc1_f)
                    if not np.isinf(co_f):
                        co_vals.append(co_f)

                except ValueError:
                    continue

    # Compute mean and std for each score
    scoap_stats["cc0"]["mean"] = np.mean(cc0_vals)
    scoap_stats["cc0"]["std"] = np.std(cc0_vals)

    scoap_stats["cc1"]["mean"] = np.mean(cc1_vals)
    scoap_stats["cc1"]["std"] = np.std(cc1_vals)

    scoap_stats["co"]["mean"] = np.mean(co_vals)
    scoap_stats["co"]["std"] = np.std(co_vals)

    return scoap_stats

def normalize_score(value, score_type):
    """
    Normalizes a given score using stored mean and std.
    score_type must be 'cc0', 'cc1', or 'co'.
    """
    if scoap_stats[score_type]["mean"] is None or scoap_stats[score_type]["std"] is None:
        raise ValueError("Stats not computed yet. Call compute_scoap_stats first.")

    mean = scoap_stats[score_type]["mean"]
    std = scoap_stats[score_type]["std"]

    if std == 0:
        return 0.0  # avoid division by zero

    # Convert to float if needed
    val = float(10000) if value == "#INF" else float(value)
    if np.isinf(val):
        return np.inf

    return (val - mean) / std

if __name__ == "__main__":
        # First, compute and store global stats
    compute_scoap_stats("Trojan_GNN/Trust-Hub/Scoap_Scores")
    print(scoap_stats)

    # Then normalize a score
    normalized_cc0 = normalize_score(15.2, "cc0")
    normalized_cc1 = normalize_score(3.7, "cc1")
    normalized_co  = normalize_score("#INF", "co")

    print("Normalized CC0:", normalized_cc0)
    print("Normalized CC1:", normalized_cc1)
    print("Normalized CO:", normalized_co)