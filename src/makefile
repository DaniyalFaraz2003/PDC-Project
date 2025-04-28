MPICXX := mpic++

SRC := main.cpp
EXEC := mpi_program

CXXFLAGS := -Wall -O3 -std=c++11

PROCS ?= 4

.PHONY: all clean run

all: $(EXEC)

$(EXEC): $(SRC)
	$(MPICXX) $(CXXFLAGS) $< -o $@

run: $(EXEC)
	mpirun -np $(PROCS) ./$(EXEC)

clean:
	rm -f $(EXEC)