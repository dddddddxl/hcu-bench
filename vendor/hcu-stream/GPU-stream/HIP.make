ifdef ROCM_PATH
    MY_ROCM := $(ROCM_PATH)
else
    MY_ROCM := /opt/rocm
endif

HIP_PATH ?= $(MY_ROCM)/hip
HIPCC=$(HIP_PATH)/bin/hipcc

ifeq ($(ARCH),)
    TARGET := $(shell $(MY_ROCM)/bin/rocminfo | grep "Name" | grep -m 1 "gfx" | awk -F':' '{print $$2}' | tr -d '[:space:]' )
else
    TARGET := $(ARCH)
    CFLAGS = -DMODEL
endif

ifneq ($(ModelTick),)
    CFLAGS += -DModelTick -I$(PMD_PATH)/src/gpu-compute -lgem5
endif

hip-stream: main.cpp HIPStream.cpp
	$(HIPCC) $(CXXFLAGS) --offload-arch=$(TARGET) -O3 -std=c++11 -DHIP $(CFLAGS) $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hip-stream

