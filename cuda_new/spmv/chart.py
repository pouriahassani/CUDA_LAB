import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("res.csv")

rows = data["Rows"]
nonzeros = data["Nonzeros"]
execution_time_coo = data["Execution Time COO (ms)"]
kernel_time_coo = data["Kernel Time COO (ms)"]
execution_time_naive = data["Execution Time Naive (ms)"]
kernel_time_naive = data["Kernel Time Naive (ms)"]

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
for nonzero in nonzeros.unique():
    subset = data[data["Nonzeros"] == nonzero]
    plt.plot(
        subset["Rows"],
        subset["Execution Time COO (ms)"],
        label=f"COO Nonzeros={nonzero}",
    )
    plt.plot(
        subset["Rows"],
        subset["Execution Time Naive (ms)"],
        label=f"Naive Nonzeros={nonzero}",
        linestyle="--",
    )

plt.xlabel("Matrix Rows/Cols")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time Comparison")
plt.legend()

plt.subplot(1, 2, 2)
for nonzero in nonzeros.unique():
    subset = data[data["Nonzeros"] == nonzero]
    plt.plot(
        subset["Rows"], subset["Kernel Time COO (ms)"], label=f"COO Nonzeros={nonzero}"
    )
    plt.plot(
        subset["Rows"],
        subset["Kernel Time Naive (ms)"],
        label=f"Naive Nonzeros={nonzero}",
        linestyle="--",
    )

plt.xlabel("Matrix Rows/Cols")
plt.ylabel("Kernel Time (ms)")
plt.title("Kernel Time Comparison")
plt.legend()

plt.tight_layout()
plt.show()
