ifdef ROCM_PATH
  MY_ROCM := $(ROCM_PATH)
else
  MY_ROCM := /opt/rocm
endif

HIPCC := $(MY_ROCM)/bin/hipcc

HIP_PATH ?= $(MY_ROCM)/hip
HIPCC = $(HIP_PATH)/bin/hipcc
HIPCC =$(MY_ROCM)/bin/hipcc

TARGET := $(shell $(MY_ROCM)/bin/rocminfo | grep "Name" | grep -m 1 "gfx" | awk -F':' '{print $$2}' | tr -d '[:space:]' )
TARGET_NUM := ${shell echo $(TARGET) | sed 's/[^0-9]//g'}
OFFLOAD_ARCH := --offload-arch=$(TARGET)
CXXFLAGS = --std=c++17 -I./include -I./kernel -I$(MY_ROCM)/include/hiprand -Wall

SRC = main.cpp src/memory/BWtestcase.cpp src/output.cpp src/device.cpp \
	src/latency/LTtestcase.cpp src/latency/Dispatchtestcase.cpp src/compute/TFtestcase.cpp \
	src/memory/L2cacheRW.cpp src/memory/TCCRead.cpp \
	src/memory/element_wise.cpp src/memory/gemmMemory.cpp \
	src/memory/faMemory.cpp src/memory/paMemory.cpp \
	src/UnitTest/VgprFlush.cpp
TARGET = hcu-stream
PROJECT_DIR := $(shell pwd)
EXTRA_FLAGS := -DDIRPATH=\"$(PROJECT_DIR)\"

# 默认目标
all: $(TARGET)

# 构建目标
$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) $(OFFLOAD_ARCH) -DHIP_ARCH=$(TARGET_NUM) $(EXTRA_FLAGS) $^ -o $@

# 清理目标
clean:
	rm -f $(TARGET)

# 伪目标
.PHONY: all clean
