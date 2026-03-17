import pandas as pd
from pathlib import Path

data_path = Path("./data/shoulder_press/shoulder_press_dennis_1.csv")
df = pd.read_csv(data_path)
print(df.head())

df.to
