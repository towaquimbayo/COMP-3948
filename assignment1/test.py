# This is only a suggestion so you can find other ways to do this.
# See example 7 in lab week 2 also for drawing multiple plots in a single row.
# Probably this code could be modularized in a function.
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(
    data={
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature": [40, 50, 50, 56, 80, 90, 100, 110, 113, 120],
    }
)

# Generate grouping column by percentile
df["group"] = pd.qcut(df["target"], q=3)
print(df)
df = pd.get_dummies(df, columns=["group"])
print(df)

dfLow = df[df["group_(0.999, 4.0]"] == 1]
dfMid = df[df["group_(4.0, 7.0]"] == 1]
dfHigh = df[df["group_(7.0, 10.0]"] == 1]

avgTargetLow = dfLow["target"].mean()
avgTargetMid = dfMid["target"].mean()
avgTargetHigh = dfHigh["target"].mean()

avgFeatureLow = dfLow["feature"].mean()
avgFeatureMid = dfMid["feature"].mean()
avgFeatureHigh = dfHigh["feature"].mean()

plt.rcParams["font.size"] = 17
plt.bar(
    [avgTargetLow, avgTargetMid, avgTargetHigh],
    [avgFeatureLow, avgFeatureMid, avgFeatureHigh],
)
plt.title("Feature vs. Target (pls show names)")
plt.ylabel("Feature Avg (show feature name)")
plt.xticks(
    ticks=[avgTargetLow, avgTargetMid, avgTargetHigh],
    labels=[avgTargetLow, avgTargetMid, avgTargetHigh],
    rotation=70,
)
plt.xlabel("Target %tile group averages (name of target)")
plt.tight_layout()
plt.show()

plt.scatter(df["target"], df["feature"])
plt.title("Feature vs. target (please show names)")
plt.xlabel("Target name")
plt.ylabel("Feature name")
plt.tight_layout()
plt.show()
