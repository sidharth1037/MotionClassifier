import pandas as pd

# Load files
acc = pd.read_csv("C:\\Users\\sidha\\Desktop\\Main Project\\basic_model\\walking\\Accelerometer.csv")
gyro = pd.read_csv("C:\\Users\\sidha\\Desktop\\Main Project\\basic_model\\walking\\Gyroscope.csv")

# Merge on "Time (s)" column
merged = pd.merge_asof(acc, gyro, on="Time (s)", direction="nearest")

# Save merged file
merged.to_csv("C:\\Users\\sidha\\Desktop\\Main Project\\basic_model\\walking\\walking12.csv", index=False)
