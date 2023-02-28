INCLUDE_DIR := $(BUILD_HOME)/include
SRC_DIR := $(BUILD_HOME)/src
BUILD_DIR := $(BUILD_HOME)/build
BIN_DIR := $(BUILD_DIR)/bin
OBJ_DIR := $(BUILD_DIR)/obj

SRC_SUBDIRS := driver subsystem1
SRC_FILES := $(wildcard $(addsuffix /*.cpp,$(SRC_SUBDIRS)))  # driver/main.cpp subsystem1/module1.cpp
OBJ_FILES := $(patsubst %.cpp,%.o,$(SRC_FILES))              # driver/main.o subsystem1/module1.o
OBJ_FILES := $(addprefix $(OBJ_DIR)/, $(OBJ_FILES))          # .../obj/driver/main.o .../obj/subsystem1/module1.o

CC := /usr/bin/g++
CXXFLAGS := -I$(INCLUDE_DIR)

all: $(BIN_DIR)/main

# ===========================
# Build binaries
# ===========================

$(BIN_DIR)/main: $(OBJ_FILES)
	mkdir -p $(@D)
	$(CC) $(CXXFLAGS) -o $@ $^

# ===========================
# Build libraries
# ===========================

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(@D)
	$(CC) $(CXXFLAGS) -c -o $@ $^

clean:
	rm -rf $(BUILD_DIR)

echoes:
	echo $(SRC_SUBDIRS)
	echo $(SRC_FILES)
	echo $(OBJ_FILES)
