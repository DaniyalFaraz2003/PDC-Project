# Compiler
CXX := g++

# Source files and executable
SRCS := main.cpp
EXEC := serial_program

# Compiler flags
CXXFLAGS := -Wall -O3 -std=c++17

.PHONY: all clean run

all: $(EXEC)

$(EXEC): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(EXEC)