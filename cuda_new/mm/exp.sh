#!/bin/bash

THREADS=(20)  # Range of thread values
SIZES=(256 512 1024 2048)  # Range of matrix sizes

# Loop over the number of threads
for NUM_THREADS in "${THREADS[@]}"; do
  # Loop over the sizes
  for SIZE in "${SIZES[@]}"; do
    make NUM_THREADS=${NUM_THREADS} SIZE=${SIZE} clean all > /dev/null
    ./build/main ${NUM_THREADS} ${SIZE}
  done
done
