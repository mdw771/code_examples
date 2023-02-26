INCLUDE_DIR := ./include
SRC_DIR := ./src
BUILD_DIR := ./build
OBJ_FILES := $(BUILD_DIR)/main.o $(BUILD_DIR)/module1.o

CC := /usr/bin/g++
CXXFLAGS := -I$(INCLUDE_DIR)

all: $(BUILD_DIR)/main.o $(BUILD_DIR)/main

# ===========================
# Build binaries
# ===========================

# n is made automatically from n.o by running the command
# $(CC) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@
$(BUILD_DIR)/main: $(BUILD_DIR)/main.o $(BUILD_DIR)/module1.o

# ===========================
# Build libraries
# ===========================

# n.o is made automatically from n.cc or n.cpp with a 
# command of the form $(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(filter %.o,$(OBJ_FILES)): $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CC) -c $(CXXFLAGS) $^ -o $@

# The command above matches the *basenames* of files in OBJ_FILES,
# and build the library for each of them. It is equivalent to the
# following: 
#
# $(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp
# 	mkdir -p $(BUILD_DIR)
# 	$(CC) -c $(CXXFLAGS) $^ -o $@
#
# $(BUILD_DIR)/module1.o: $(SRC_DIR)/module1.cpp
# 	mkdir -p $(BUILD_DIR)
# 	$(CC) -c $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)