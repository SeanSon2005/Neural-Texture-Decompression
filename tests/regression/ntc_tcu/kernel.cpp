// Tensor neural texture decoder: one warp decodes one tile of up to 8 same-material
// requests with int8 WMMA (8 x 8 x 32 tiles at NUM_THREADS = 8).
//
// Fragment layout (NT = 8, from wmma_context): lane l, register r = m * 4 + q holds
//   A: row l/2 + 4m, K word (l&1) + 2q          (4 int8 features = one TEX word)
//   C: row l/2 + 4m, column (l&1) + 2q          (one int32)
// Rows match, columns do not: a lane holds every other output column, but the next
// layer's A wants 4 consecutive inputs per register. The host stores each layer's output
// rows permuted within groups of 8 (physical column p computes neuron 4*(p&1) + (p>>1)),
// so register (m, j) of N-tile t holds neuron 8t + 4*(l&1) + j: exactly byte j of A
// register (m, t%4) in K-tile t/4. Requantize + repack is therefore lane-local, with no
// shared-memory round trip between layers.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_graphics.h>
#include <vx_intrinsics.h>
#include "ntc_common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::int8, vt::int32>;
using cfg = vt::wmma_config_t<VX_CFG_NUM_THREADS, vt::fp32, vt::fp32>;
using vt::detail::unroll_for;

static_assert(ctx::tileM == NTC_TILE_M && ctx::tileN == 8 && ctx::tileK == NTC_FEATS,
              "decoder assumes NUM_THREADS=8 (an 8x8x32 int8 tile)");
static_assert(cfg::tcM == 4 && cfg::tcN == 2 && cfg::tcK == 2 && cfg::m_steps == 2 &&
              cfg::n_steps == 4 && cfg::k_steps == 4 && cfg::a_sub_blocks == 1 &&
              ctx::fragment_a::NR == 8 && ctx::fragment_acc::NR == 8,
              "lane-local repack assumes the NT=8 fragment layout");

static inline __attribute__((always_inline)) float as_f(uint32_t x) { return __builtin_bit_cast(float, x); }
static inline __attribute__((always_inline)) int32_t as_i(float x) { return __builtin_bit_cast(int32_t, x); }

// acc = bias (same for every row) + A x W^T over KT K-tiles, for N-tile t.
template <uint32_t KT>
static inline __attribute__((always_inline))
void tile_mma(ctx::fragment_acc& c, const ctx::fragment_a (&a)[KT],
              const int8_t* W, const int32_t* bias, uint32_t ldm, uint32_t t) {
  ctx::load_matrix_sync(c, bias + 8 * t, 0);   // ldm 0 broadcasts the bias row
  unroll_for<KT>([&](auto kt) {
    ctx::fragment_b b;
    ctx::load_matrix_sync<vt::col_major>(b, W + 8 * t * ldm + 32 * kt, ldm);
    ctx::mma_sync(c, a[kt], b, c);
  });
}

// Requantize accumulator row-block m (4 int32) into one packed word.
static inline __attribute__((always_inline))
uint32_t pack_row(const ctx::fragment_acc& c, uint32_t m, uint32_t shift, int32_t hi) {
  uint32_t w = 0;
  unroll_for<cfg::n_steps>([&](auto j) {
    w |= ntc_requant(as_i(c.data[m * cfg::n_steps + j]), shift, hi) << (8 * j);
  });
  return w;
}

// Hidden layer: KT input K-tiles -> H/8 output N-tiles, written as the next layer's A.
template <uint32_t KT, uint32_t NT>
static inline __attribute__((always_inline))
void hidden_layer(const ctx::fragment_a (&a)[KT], ctx::fragment_a (&next)[(NT + 3) / 4],
                  const uint8_t* blob, const ntc_layer_t& L) {
  auto W = reinterpret_cast<const int8_t*>(blob + L.w_off);
  auto B = reinterpret_cast<const int32_t*>(blob + L.b_off);
  unroll_for<(NT + 3) / 4>([&](auto f) { ctx::fill_fragment(next[f], 0); });  // K padding
  unroll_for<NT>([&](auto t) {
    ctx::fragment_acc c;
    tile_mma<KT>(c, a, W, B, L.ldm, t);
    unroll_for<cfg::m_steps>([&](auto m) {
      next[t / 4].data[m * cfg::k_steps + t % 4] = as_f(pack_row(c, m, L.shift, 127));
    });
  });
}

template <uint32_t H>
static inline __attribute__((always_inline))
void decode_tile(const kernel_arg_t* arg, const ntc_tile_t* tile, uint32_t lane) {
  constexpr uint32_t KT = (H + 31) / 32;   // K-tiles of a hidden-layer input
  constexpr uint32_t NT = H / 8;           // N-tiles of a hidden-layer output
  auto req    = reinterpret_cast<const uint32_t*>(arg->req_addr);
  auto origin = reinterpret_cast<const uint32_t*>(arg->mat_addr);
  auto blob   = reinterpret_cast<const uint8_t*>(arg->wt_addr);
  auto out    = reinterpret_cast<uint32_t*>(arg->out_addr);
  uint32_t material = tile->material;
  uint32_t count    = tile->count;
  auto L = reinterpret_cast<const ntc_layer_t*>(arg->layer_addr) + material * NTC_LAYERS;

  // This lane's two tile rows (m = 0, 1).
  uint32_t ua[cfg::m_steps], va[cfg::m_steps], rid[cfg::m_steps];
  unroll_for<cfg::m_steps>([&](auto m) {
    rid[m] = tile->rows[lane / cfg::tcK + m * cfg::tcM];
    ntc_atlas_uv(arg, origin, material, req[3 * rid[m] + 1], req[3 * rid[m] + 2], &ua[m], &va[m]);
  });

  // Layer-1 A straight from TEX: register r holds TEX word (l&1) + 2*(r%4) of row m = r/4.
  // The stage (word >> 2) depends only on r, so it stays an immediate.
  ctx::fragment_a x[1];
  unroll_for<ctx::fragment_a::NR>([&](auto r) {
    constexpr uint32_t m       = r / cfg::k_steps;
    constexpr uint32_t block_k = r % cfg::k_steps;
    constexpr uint32_t stage   = block_k >> 1;
    uint32_t mip = (lane % cfg::tcK + block_k * cfg::tcK) & 3;
    uint32_t texel = vortex::graphics::vx_tex(stage, ua[m], va[m], mip);
    x[0].data[r] = as_f(texel ^ 0x80808080u);   // u8 -> i8 (byte - 128)
  });

  ctx::fragment_a h1[KT], h2[KT];
  hidden_layer<1, NT>(x, h1, blob, L[0]);
  hidden_layer<KT, NT>(h1, h2, blob, L[1]);

  // Output layer: one N-tile; register (m, j) = channel 4*(l&1) + j of row m.
  ctx::fragment_acc c;
  tile_mma<KT>(c, h2, reinterpret_cast<const int8_t*>(blob + L[2].w_off),
               reinterpret_cast<const int32_t*>(blob + L[2].b_off), L[2].ldm, 0);
  unroll_for<cfg::m_steps>([&](auto m) {
    uint32_t w = pack_row(c, m, L[2].shift, 255);
    if (lane / cfg::tcK + m * cfg::tcM < count)
      out[2 * rid[m] + (lane & 1)] = w;
  });
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto tile = reinterpret_cast<const ntc_tile_t*>(arg->tile_addr) + blockIdx.x;
  uint32_t lane = vx_thread_id();
  switch (arg->hidden) {
  case 16: decode_tile<16>(arg, tile, lane); break;
  case 32: decode_tile<32>(arg, tile, lane); break;
  case 64: decode_tile<64>(arg, tile, lane); break;
  default: break;
  }
}
