import pandas as pd

rows = []
with open("gesture_data.csv", "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 43:  # 42 coordinates + 1 label
            rows.append(parts)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save cleaned CSV
df.to_csv("gesture_data_clean.csv", index=False, header=False)
print("Cleaned CSV saved as gesture_data_clean.csv")
