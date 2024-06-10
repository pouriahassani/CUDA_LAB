#!/bin/bash

ROWS=(128 256 512 1024)  # Number of rows and columns (square matrix)
NONZEROS=(1000 5000 10000 50000)  # Number of non-zero elements

echo "Rows,Cols,Nonzeros,Execution Time COO (ms),Kernel Time COO (ms),Execution Time Naive (ms),Kernel Time Naive (ms)" > res.csv

for ROW in "${ROWS[@]}"; do
  for NONZERO in "${NONZEROS[@]}"; do
    echo "Running with Rows=${ROW}, Cols=${ROW}, Nonzeros=${NONZERO}"
    ./build/main ${ROW} ${ROW} ${NONZERO}
  done
done
