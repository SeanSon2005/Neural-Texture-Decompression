// Scalar neural texture decoder: one thread per request, lanes may hold different
// materials. Same feature lookup as ntc_tcu (8 vx_tex per request); the MLP runs on the
// integer ALU with the weights read from memory.

#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_intrinsics.h>
#include "ntc_common.h"

#define NTC_MAX_HIDDEN 64

// y[n] = requant(b[n] + sum_k W[n][k] * x[k]) for n < out.
static inline __attribute__((always_inline))
void dense(const int32_t* x, uint32_t in, int32_t* y, uint32_t out,
           const uint8_t* blob, const ntc_layer_t& L, int32_t hi) {
  auto w = reinterpret_cast<const int8_t*>(blob + L.w_off);
  auto b = reinterpret_cast<const int32_t*>(blob + L.b_off);
  for (uint32_t n = 0; n < out; ++n) {
    const int8_t* row = w + n * L.ldm;
    int32_t acc = b[n];
    for (uint32_t k = 0; k < in; ++k) acc += (int32_t)row[k] * x[k];
    y[n] = (int32_t)ntc_requant(acc, L.shift, hi);
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < arg->num_items) {
    auto req    = reinterpret_cast<const uint32_t*>(arg->req_addr) + 3 * i;
    auto origin = reinterpret_cast<const uint32_t*>(arg->mat_addr);
    auto table  = reinterpret_cast<const ntc_layer_t*>(arg->layer_addr);
    auto blob   = reinterpret_cast<const uint8_t*>(arg->wt_addr);
    auto out    = reinterpret_cast<uint32_t*>(arg->out_addr);

    uint32_t material = req[0];
    uint32_t ua, va;
    ntc_atlas_uv(arg, origin, material, req[1], req[2], &ua, &va);

    // word = stage * 4 + mip; the stage is an immediate in the instruction.
    uint32_t words[NTC_WORDS];
    words[0] = vortex::graphics::vx_tex(0, ua, va, 0);
    words[1] = vortex::graphics::vx_tex(0, ua, va, 1);
    words[2] = vortex::graphics::vx_tex(0, ua, va, 2);
    words[3] = vortex::graphics::vx_tex(0, ua, va, 3);
    words[4] = vortex::graphics::vx_tex(1, ua, va, 0);
    words[5] = vortex::graphics::vx_tex(1, ua, va, 1);
    words[6] = vortex::graphics::vx_tex(1, ua, va, 2);
    words[7] = vortex::graphics::vx_tex(1, ua, va, 3);

    int32_t x[NTC_FEATS];
    for (uint32_t k = 0; k < NTC_FEATS; ++k)
      x[k] = (int32_t)((words[k >> 2] >> (8 * (k & 3))) & 0xff) - 128;

    const ntc_layer_t* L = table + material * NTC_LAYERS;
    uint32_t H = arg->hidden;
    int32_t h1[NTC_MAX_HIDDEN], h2[NTC_MAX_HIDDEN], y[NTC_CHANNELS];
    dense(x,  NTC_FEATS, h1, H, blob, L[0], 127);
    dense(h1, H,         h2, H, blob, L[1], 127);
    dense(h2, H,         y,  NTC_CHANNELS, blob, L[2], 255);

    out[2 * i]     = y[0] | (y[1] << 8) | (y[2] << 16) | ((uint32_t)y[3] << 24);
    out[2 * i + 1] = y[4] | (y[5] << 8) | (y[6] << 16) | ((uint32_t)y[7] << 24);
  }
}
