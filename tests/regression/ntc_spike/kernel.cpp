#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_graphics.h>
#include <vx_intrinsics.h>
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::int8, vt::int32>;
// wmma_context's own register layout (it sizes every config from an fp32 view).
using cfg = vt::wmma_config_t<VX_CFG_NUM_THREADS, vt::fp32, vt::fp32>;

static_assert(ctx::tileM == NTC_REQS && ctx::tileN == NTC_OUTS && ctx::tileK == NTC_FEATS,
              "spike assumes NUM_THREADS=8 (an 8x8x32 int8 tile)");
// The TEX stage rides funct2, so it must be a compile-time constant per A register.
// With tcK == 2 and k_steps == 4 the lane only picks the low bit of the word index,
// which lands in the mip level, never the stage.
static_assert(cfg::tcK == 2 && cfg::k_steps == 4 && cfg::a_sub_blocks == 1,
              "A-fragment -> (request, word) mapping below assumes the NT=8 layout");

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto uv   = reinterpret_cast<const uint32_t*>(arg->uv_addr);
  auto W    = reinterpret_cast<const ctx::input_t*>(arg->w_addr);
  auto feat = reinterpret_cast<uint32_t*>(arg->feat_addr);
  auto Y    = reinterpret_cast<ctx::output_t*>(arg->y_addr);

  uint32_t lane = vx_thread_id();

  // Fill fragment A straight from TEX results, mirroring load_matrix_sync's
  // row-major A layout: register r of this lane holds 4 int8 features of one
  // (request row, texel word). No memory round trip.
  ctx::fragment_a fragA;
  vt::detail::unroll_for<ctx::fragment_a::NR>([&](auto r) {
    constexpr uint32_t block_m = r / cfg::k_steps;
    constexpr uint32_t block_k = r % cfg::k_steps;
    constexpr uint32_t stage   = block_k >> 1;
    uint32_t row  = lane / cfg::tcK + block_m * cfg::tcM;
    uint32_t word = lane % cfg::tcK + block_k * cfg::tcK;   // = stage * 4 + mip
    uint32_t mip  = word & 3;
    uint32_t texel = vortex::graphics::vx_tex(stage, uv[2 * row], uv[2 * row + 1], mip);
    feat[row * NTC_WORDS + word] = texel;
    uint32_t a_word = texel ^ 0x80808080u;                  // u8 -> i8 (x - 128)
    fragA.data[r] = __builtin_bit_cast(float, a_word);
  });

  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);
  ctx::load_matrix_sync<vt::col_major>(fragB, W, NTC_FEATS);
  ctx::mma_sync(fragC, fragA, fragB, fragC);
  ctx::store_matrix_sync(Y, fragC, NTC_OUTS);
}
