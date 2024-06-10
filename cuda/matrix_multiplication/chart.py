import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("mm.csv")

plt.figure(figsize=(10, 5))
# plt.plot(data["Size"], data["CPU Time(ms)"], label="CPU Time")
plt.plot(data["Size"], data["CUDA Time(ms)"], label="CUDA Time")
plt.plot(data["Size"], data["Tiled CUDA Time(ms)"], label="Tiled CUDA Time")
plt.xlabel("Matrix Size")
plt.ylabel("Time (ms)")
plt.title("Matrix Multiplication Benchmark")
plt.legend()
plt.grid(True)
plt.show()
