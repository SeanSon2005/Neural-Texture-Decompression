ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# TEX feeds the TCU directly, so both units and int8 tiles are required.
NTC_CONFIGS := -DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_NUM_THREADS=8
CONFIGS := $(CONFIGS) $(filter-out $(CONFIGS),$(NTC_CONFIGS))

PROJECT := ntc_spike

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

KERNEL_LIB := vortex2

OPTS ?=

include ../common.mk
