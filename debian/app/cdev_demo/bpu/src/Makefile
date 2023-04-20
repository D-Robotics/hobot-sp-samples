# Pre-compiler and Compiler flags
CXX_FLAGS :=  -std=c++14 -march=armv8-a -mcpu=cortex-a53 -O3
PRE_FLAGS := 

# Project directory structure
BIN := bin
SRC := .
LIB := 
INC := ../include /usr/include /usr/include/vio/
MAINFILE := $(SRC)/hb_dnn_test.cpp

# Build directories and output
TARGET := $(BIN)/sample
BUILD := build
# Library search directories and flags
EXT_LIB :=/usr/lib /usr/lib/hbbpu/
LDFLAGS :=-ldnn -lpthread -lspcdev -lopencv_world
LDPATHS := $(addprefix -L,$(LIB) $(EXT_LIB))

# Include directories
INC_DIRS := $(INC) $(shell find $(SRC) -type d) 
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Construct build output and dependency filenames
SRCS := $(shell find $(SRC) -name "*.cpp")
OBJS := $(subst $(SRC)/,$(BUILD)/,$(addsuffix .o,$(basename $(SRCS))))
DEPS := $(OBJS:.o=.d)



# Build task
build: all

# Main task
all: $(TARGET)

# Task producing target from built files
$(TARGET): $(OBJS)
	@echo "Building..."
	$(CXX) $(CXX_FLAGS) $(OBJS) -o $@ $(LDPATHS) $(LDFLAGS)

# Compile all cpp files
$(BUILD)/%.o: $(SRC)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(PRE_FLAGS) $(INC_FLAGS) -c -o $@ $<

# Clean task
.PHONY: clean
clean:
	@echo "Clearing..."
	rm -rf build

# Include all dependencies
-include $(DEPS)
