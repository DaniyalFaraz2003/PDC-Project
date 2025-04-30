# Compiler
MPICXX := mpic++

# Source files and executable
SRCS := main.cpp
EXEC := mpi_program

# Compiler flags
CXXFLAGS := -Wall -O3 -std=c++17

# Default number of processes
PROCS ?= 4

.PHONY: all clean run

all: $(EXEC)

$(EXEC): $(SRCS)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

run: $(EXEC)
	mpirun -np $(PROCS) ./$(EXEC)

clean:
	rm -f $(EXEC)