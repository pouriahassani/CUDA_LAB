import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("res.csv")

size = data["Size"]
cpu_time = data["CPU Time(ms)"]
cuda_time = data["CUDA Time(ms)"]
tiled_cuda_time = data["Tiled CUDA Time(ms)"]

plt.figure(figsize=(10, 6))

plt.plot(size, cpu_time, label="CPU Time(ms)", marker="o")
plt.plot(size, cuda_time, label="CUDA Time(ms)", marker="o")
plt.plot(size, tiled_cuda_time, label="Tiled CUDA Time(ms)", marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("Time (ms)")
plt.title("Performance Comparison")
plt.legend()
plt.grid(True)
plt.yscale("log")

plt.show()
