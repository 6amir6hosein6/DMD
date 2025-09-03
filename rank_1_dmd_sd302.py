
### true_match_dmd

import pandas as pd
import numpy as np
import re

def extract_id(filename):
    if not isinstance(filename, str):
        return None
    match = re.match(r"(\d{8})", filename)
    if match:
        return match.group(1)
    return None

# Load score matrix
df = pd.read_csv(
    "/home/ubuntu5/amirhossein/DMD/TEST_DATA/NIST27/DMD_6/score_matrix_relax_sorted.csv",
    index_col=0
)

# Load GT matches (add .pkl to match CSV format)
match_dict = {}
with open("/home/ubuntu5/amirhossein/DMD/datasets/N2NLatent_genuine_pairs_edit.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or "," not in line:
            continue
        latent, correct_visible = line.split(",", 1)
        latent = latent.strip() + ".pkl"
        correct_visible = correct_visible.strip() + ".pkl"
        match_dict[latent] = correct_visible

match_count = 0
total_count = 0
results = []

for latent_file, row in df.iterrows():
    if latent_file not in match_dict:
        continue
    
    correct_visible = match_dict[latent_file]
    scores = row.values.astype(float)
    max_idx = np.argmax(scores)
    predicted_visible = df.columns[max_idx].split(",")[0].strip()
    max_score = scores[max_idx]

    is_match = (predicted_visible == correct_visible)
    if is_match:
        match_count += 1
    total_count += 1

    results.append(f"latent: {latent_file}")
    results.append(f"predicted_visible: {predicted_visible}")
    results.append(f"correct_visible: {correct_visible}")
    results.append(f"score: {max_score:.4f}")
    results.append(f"match: {is_match}")
    results.append("---")

match_percent = (match_count / total_count) * 100 if total_count > 0 else 0

with open("score_matrix_relax_results_full_sd302latent_mntverifinger_dmdraw_sample_gpu.txt", "w") as f:
    f.write(f"Rank-1 Match Rate (based on GT file): {match_percent:.2f}%\n")
    f.write("=" * 40 + "\n")
    f.write("\n".join(results))

print(f" Rank-1 Accuracy: {match_percent:.2f}% ")

