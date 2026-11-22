#ifndef _NTC_COMMON_H_
#define _NTC_COMMON_H_

// Shared by the ntc_scalar and ntc_tcu decoders (host and kernel).
// The integer decode contract is ntc/quant.py; the atlas mapping is ntc/atlas.py.

#include <stdint.h>

#define NTC_STAGES   2   // TEX stages = feature pyramids
#define NTC_MIPS     4   // mip levels per pyramid
#define NTC_WORDS    (NTC_STAGES * NTC_MIPS)   // texel words per request, word = stage * 4 + mip
#define NTC_FEATS    (NTC_WORDS * 4)           // int8 features per request
#define NTC_CHANNELS 8   // decoded bytes per request
#define NTC_LAYERS   3   // 32 -> H -> H -> 8
#define NTC_TILE_M   8   // requests per WMMA tile (NUM_THREADS = 8)

// One layer of one material. Offsets are bytes into the weight blob.
// Scalar blob: W int8 [out][in] row-major (exactly weights.bin).
// TCU blob:    W int8 [out_pad][in_pad], rows permuted within each 8-row group
//              (see ntc_tcu/kernel.cpp), bias int32 [out_pad] permuted the same way.
typedef struct {
  uint32_t w_off;
  uint32_t b_off;
  uint32_t shift;
  uint32_t ldm;     // row stride of W in bytes (in, or in_pad for the TCU blob)
} ntc_layer_t;

// One WMMA tile: up to 8 requests of the same material. rows[i] for i >= count
// are padding and repeat rows[0] so their TEX lookups stay in range.
typedef struct {
  uint32_t material;
  uint32_t count;
  uint32_t rows[NTC_TILE_M];
} ntc_tile_t;

typedef struct {
  uint64_t req_addr;     // uint32 [n][3]: material, u_local, v_local (TEX fixed point)
  uint64_t tile_addr;    // ntc_tile_t [num_items]          (TCU only)
  uint64_t mat_addr;     // uint32 [materials][2]: atlas origin u, v
  uint64_t layer_addr;   // ntc_layer_t [materials][NTC_LAYERS]
  uint64_t wt_addr;      // weight blob
  uint64_t out_addr;     // uint32 [n][2]: byte c of the pair = channel c
  uint32_t num_items;    // requests (scalar) or tiles (TCU)
  uint32_t hidden;
  uint32_t log_tiles_x;
  uint32_t log_tiles_y;
  uint32_t local_min;
  uint32_t local_max;
} kernel_arg_t;

#ifdef __VORTEX__
// Material-local -> atlas fixed point; identical to AtlasLayout.atlas_uv.
static inline __attribute__((always_inline))
void ntc_atlas_uv(const kernel_arg_t* arg, const uint32_t* origin, uint32_t material,
                  uint32_t u, uint32_t v, uint32_t* ua, uint32_t* va) {
  u = u < arg->local_min ? arg->local_min : (u > arg->local_max ? arg->local_max : u);
  v = v < arg->local_min ? arg->local_min : (v > arg->local_max ? arg->local_max : v);
  *ua = origin[2 * material]     + (u >> arg->log_tiles_x);
  *va = origin[2 * material + 1] + (v >> arg->log_tiles_y);
}

// clamp((acc + 2^(s-1)) >> s, 0, hi), arithmetic shift.
static inline __attribute__((always_inline)) uint32_t ntc_requant(int32_t acc, uint32_t s, int32_t hi) {
  int32_t q = (acc + (1 << (s - 1))) >> s;
  q = q < 0 ? 0 : q;
  return (uint32_t)(q > hi ? hi : q);
}
#endif

#endif
