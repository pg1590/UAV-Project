import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- LOAD CSV ----------------
df = pd.read_csv("logs/circle.csv")

t = df["time"]

# ---------------- CAMERA PARAMETERS ----------------
FOCAL_LENGTH = 466  # same used in controller

# ---------------- PIXEL → METERS ----------------
df["error_x_m"] = (df["depth"] * df["error_x"]) / FOCAL_LENGTH
df["error_y_m"] = (df["depth"] * df["error_y"]) / FOCAL_LENGTH

# z error already in meters
df["error_z_m"] = df["error_z"]

# ---------------- ERROR PLOTS ----------------
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(t, df["error_x_m"], label="X error")
plt.ylabel("Error (m)")
plt.title("Tracking Errors in Meters")
plt.grid()

plt.subplot(3,1,2)
plt.plot(t, df["error_y_m"], label="Y error", color='orange')
plt.ylabel("Error (m)")
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, df["error_z_m"], label="Z error", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.grid()

plt.tight_layout()
plt.show()

# ---------------- DEPTH ----------------
plt.figure(figsize=(8,5))
plt.plot(t, df["depth"])
plt.xlabel("Time (s)")
plt.ylabel("Depth (m)")
plt.title("Depth vs Time")
plt.grid()
plt.show()

# ---------------- CONTROL COMMANDS ----------------
plt.figure(figsize=(10,6))

plt.plot(t, df["yaw_cmd"], label="Yaw")
plt.plot(t, df["forward_cmd"], label="Forward")
plt.plot(t, df["up_cmd"], label="Up")

plt.xlabel("Time (s)")
plt.ylabel("Control Signal")
plt.title("PID Control Commands")

plt.legend()
plt.grid()
plt.show()

# ---------------- ERROR NORM ----------------
error_norm = np.sqrt(
    df["error_x_m"]**2 +
    df["error_y_m"]**2 +
    df["error_z_m"]**2
)

plt.figure(figsize=(8,5))
plt.plot(t, error_norm)

plt.xlabel("Time (s)")
plt.ylabel("Total Error (m)")
plt.title("Total Tracking Error Norm")

plt.grid()
plt.show()

# ---------------- TRACKING STATUS ----------------
plt.figure(figsize=(8,3))
plt.plot(t, df["tracking"])

plt.xlabel("Time (s)")
plt.ylabel("Tracking (1/0)")
plt.title("Tracking Stability")

plt.grid()
plt.show()