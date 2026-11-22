ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# TEX feeds int8 WMMA tiles directly.
NTC_CONFIGS := -DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_NUM_THREADS=8
CONFIGS := $(CONFIGS) $(filter-out $(CONFIGS),$(NTC_CONFIGS))

PROJECT := ntc_tcu

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
NTC_COMMON := $(VORTEX_HOME)/tests/regression/ntc_common

SRCS := $(NTC_COMMON)/host.cpp
HDRS := $(NTC_COMMON)/ntc_common.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(NTC_COMMON)/ntc_common.h

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -I$(NTC_COMMON)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -I$(NTC_COMMON) -DNTC_TCU=1

KERNEL_LIB := vortex2

# -e <export dir> is required; see ntc_common/host.cpp for the other flags.
OPTS ?=

include ../common.mk
