#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// One warp decodes one tile: 8 requests x 32 int8 features -> 8 int32 outputs.
#define NTC_REQS     8   // tile rows (M)
#define NTC_WORDS    8   // texel words per request: 2 TEX stages x 4 mip levels
#define NTC_OUTS     8   // tile columns (N)
#define NTC_FEATS    (NTC_WORDS * 4)

typedef struct {
  uint64_t uv_addr;    // uint32_t[NTC_REQS][2]: TEX_FXD_FRAC fixed-point u, v
  uint64_t w_addr;     // int8_t[NTC_OUTS][NTC_FEATS], row = one output's weights
  uint64_t feat_addr;  // uint32_t[NTC_REQS][NTC_WORDS]: raw TEX results, for checking
  uint64_t y_addr;     // int32_t[NTC_REQS][NTC_OUTS]
} kernel_arg_t;

#endif
