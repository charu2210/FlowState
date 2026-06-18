import pandas as pd

df = pd.read_csv("data/features_live.csv")

n = len(df)

rest = int(0.33 * n)
fatigue = int(0.33 * n)

labels = [1]*rest + [0]*(n - rest - fatigue) + [-1]*fatigue

df["label"] = labels
df = df[df["label"] != 0]

df.to_csv("data/labeled_sessions.csv", index=False)

print("✅ labeled_sessions.csv created")