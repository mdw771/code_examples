INCLUDE_DIR := /mnt/d/Research/Programs/code_examples/cpp/makefile/include
BUILD_DIR := ./build

CC := /usr/bin/g++
CXXFLAGS := -I$(INCLUDE_DIR)

all: main

# ===========================
# Build binaries
# ===========================

main: main.o module1.o

# ===========================
# Build libraries
# ===========================

main.o: main.cpp
	echo $(CXXFLAGS)
	$(CC) $(CXXFLAGS) $^ -o $@

module1.o: module1.cpp
	$(CC) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)