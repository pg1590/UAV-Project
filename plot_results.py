import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("flight_log.csv")

t = df["time"]

# ---------------- ERROR PLOTS ----------------
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(t, df["error_x"], label="X error")
plt.ylabel("Error (px)")
plt.title("Tracking Errors")
plt.grid()

plt.subplot(3,1,2)
plt.plot(t, df["error_y"], label="Y error", color='orange')
plt.ylabel("Error (px)")
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, df["error_z"], label="Z error", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.grid()

plt.tight_layout()
plt.show()


# ---------------- DEPTH ----------------
plt.figure()
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
plt.ylabel("Control")
plt.title("Control Commands")
plt.legend()
plt.grid()
plt.show()


# ---------------- ERROR NORM ----------------
error_norm = np.sqrt(
    df["error_x"]**2 +
    df["error_y"]**2 +
    df["error_z"]**2
)

plt.figure()
plt.plot(t, error_norm)
plt.xlabel("Time (s)")
plt.ylabel("Error Norm")
plt.title("Total Tracking Error")
plt.grid()
plt.show()


# ---------------- TRACKING STATUS ----------------
plt.figure()
plt.plot(t, df["tracking"])
plt.xlabel("Time (s)")
plt.ylabel("Tracking (1/0)")
plt.title("Tracking Stability")
plt.grid()
plt.show()