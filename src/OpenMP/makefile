# Compiler
MPICXX := mpic++

# Source files and executable
SRCS := main.cpp
EXEC := omp_program

# Compiler flags with OpenMP support
CXXFLAGS := -Wall -O3 -std=c++17 -fopenmp

# Default number of processes
PROCS ?= 4

# Default number of OpenMP threads per process
THREADS ?= 128

.PHONY: all clean run

all: $(EXEC)

$(EXEC): $(SRCS)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

run: $(EXEC)
	export OMP_NUM_THREADS=$(THREADS) && \
	mpirun -np $(PROCS) ./$(EXEC)

clean:
	rm -f $(EXEC)