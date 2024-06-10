NVCC = nvcc
NUM_THREADS ?= 2
SIZE ?= 256

CFLAGS = -DNUM_THREADS=$(NUM_THREADS) -DSIZE=$(SIZE)
NVCCFLAGS = -Xcompiler "-fopenmp" $(CFLAGS)
LDFLAGS = -fopenmp
CUDA_LDFLAGS = -lcuda -lcudart

BIN = main

all: clean ./build/$(BIN)

run: ./build/$(BIN)
	@$<

./build/$(BIN): ./build/main.o
	@echo "Linking and generating the binary executable \"$(BIN)\""
	$(NVCC) $< -o $@ $(CUDA_LDFLAGS)

	@echo ""
	@echo "The binary file is located at ./build/$(BIN)"

./build/%.o: ./src/%.cu
	@mkdir -p ./build
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	@echo "Cleaning"
	@echo "Deleting the build directory."
	@rm -fr ./build
