// De-risking spike for neural texture decoding:
//   (a) TEX and TCU enabled in the same build and kernel,
//   (b) vx_tex results placed directly into an int8 WMMA A fragment,
//   (c) int8 x int8 -> int32 WMMA bit-exact against a host reference,
//   (d) per-mip bilinear samples bit-exact against the host TEX model.
// Two TEX stages each hold a 4-level feature pyramid of random A8R8G8B8 texels.

#include <iostream>
#include <vector>
#include <random>
#include <unistd.h>
#include <vortex2.h>
#include <VX_types.h>
#include <vx_gfx_abi.h>
#include <gfx_frag_tex.h>
#include "common.h"

using namespace vortex;

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

static const uint32_t kMips = 4;
static const uint32_t kStages = 2;
static const uint32_t kLogDim[kStages] = {6, 5};  // stage 0: 64x64, stage 1: 32x32

const char* kernel_file = "kernel.vxbin";
uint32_t    seed        = 1;
uint32_t    filter      = VX_TEX_FILTER_BILINEAR;
uint32_t    wrap        = VX_TEX_WRAP_CLAMP;

vx_device_h device  = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
vx_buffer_h tex_buf[kStages] = {};
vx_buffer_h uv_buf = nullptr, w_buf = nullptr, feat_buf = nullptr, y_buf = nullptr;

static void cleanup() {
  if (!device) return;
  for (auto b : tex_buf) if (b) vx_buffer_release(b);
  for (auto b : {uv_buf, w_buf, feat_buf, y_buf}) if (b) vx_buffer_release(b);
  if (kernel)  vx_kernel_release(kernel);
  if (module_) vx_module_release(module_);
  if (queue)   vx_queue_release(queue);
  vx_device_dump_perf(device, stdout);
  vx_device_release(device);
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:s:g:w:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 's': seed = atoi(optarg); break;
    case 'g': filter = atoi(optarg); break;
    case 'w': wrap = atoi(optarg); break;
    default:
      std::cout << "Usage: [-k kernel] [-s seed] [-g filter 0=point 1=bilinear] [-w wrap]" << std::endl;
      exit(c == 'h' ? 0 : -1);
    }
  }
}

// A mip chain laid out the way the TEX unit addresses it: level l is
// (dim >> l)^2 texels, mip offsets in bytes from the stage base.
struct Pyramid {
  std::vector<uint32_t> texels;
  uint32_t mip_off[kMips];
};

static Pyramid make_pyramid(uint32_t log_dim, std::mt19937& rng) {
  Pyramid p;
  uint32_t n = 0;
  for (uint32_t l = 0; l < kMips; ++l) {
    p.mip_off[l] = n * 4;
    uint32_t d = 1u << (log_dim - l);
    n += d * d;
  }
  p.texels.resize(n);
  for (auto& t : p.texels) t = rng();
  return p;
}

static uint32_t ref_sample(const Pyramid& p, uint32_t log_dim, uint32_t u, uint32_t v, uint32_t mip) {
  uint32_t logdim = (log_dim << 16) | log_dim;
  auto req = gfx_tex::tex_compute_request(p.mip_off[mip], logdim, VX_TEX_FORMAT_A8R8G8B8,
                                          filter, (wrap << 16) | wrap, (int32_t)u, (int32_t)v, mip);
  uint32_t texels[4] = {};
  uint32_t taps = (filter == VX_TEX_FILTER_BILINEAR) ? 4 : 1;
  for (uint32_t i = 0; i < taps; ++i) texels[i] = p.texels[req.addr[i] / 4];
  return gfx_tex::tex_apply_filter(req, texels);
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::mt19937 rng(seed);

  Pyramid pyr[kStages];
  for (uint32_t s = 0; s < kStages; ++s) pyr[s] = make_pyramid(kLogDim[s], rng);

  const uint32_t one = 1u << TEX_FXD_FRAC;
  std::vector<uint32_t> uv(NTC_REQS * 2);
  for (auto& x : uv) x = rng() % one;
  std::vector<int8_t> W(NTC_OUTS * NTC_FEATS);
  for (auto& w : W) w = (int8_t)(rng() & 0xff);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  if (!(isa_flags & VX_ISA_EXT_TEX) || !(isa_flags & VX_ISA_EXT_TCU)) {
    std::cout << "Error: needs both TEX and TCU (spike a)" << std::endl;
    cleanup();
    return -1;
  }
  if (num_threads != 8) {
    std::cout << "Error: needs NUM_THREADS=8, got " << num_threads << std::endl;
    cleanup();
    return -1;
  }

  // TEX reads through physical addresses (bypasses the per-core MMU).
  for (uint32_t s = 0; s < kStages; ++s) {
    uint64_t addr;
    uint32_t bytes = pyr[s].texels.size() * 4;
    RT_CHECK(vx_buffer_create(device, bytes, VX_MEM_READ | VX_MEM_PHYS, &tex_buf[s]));
    RT_CHECK(vx_buffer_address(tex_buf[s], &addr));
    RT_CHECK(vx_enqueue_write(queue, tex_buf[s], 0, pyr[s].texels.data(), bytes, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_STAGE,  s, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_LOGDIM, (kLogDim[s] << 16) | kLogDim[s], 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_FORMAT, VX_TEX_FORMAT_A8R8G8B8, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_WRAP,   (wrap << 16) | wrap, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_FILTER, filter, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_ADDR,   addr / 64, 0, nullptr, nullptr));
    for (uint32_t l = 0; l <= (uint32_t)VX_TEX_LOD_MAX; ++l) {
      uint32_t off = pyr[s].mip_off[l < kMips ? l : kMips - 1];
      RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_MIPOFF_BASE + l, off, 0, nullptr, nullptr));
    }
  }

  kernel_arg_t arg = {};
  RT_CHECK(vx_buffer_create(device, uv.size() * 4, VX_MEM_READ, &uv_buf));
  RT_CHECK(vx_buffer_create(device, W.size(), VX_MEM_READ, &w_buf));
  RT_CHECK(vx_buffer_create(device, NTC_REQS * NTC_WORDS * 4, VX_MEM_WRITE, &feat_buf));
  RT_CHECK(vx_buffer_create(device, NTC_REQS * NTC_OUTS * 4, VX_MEM_WRITE, &y_buf));
  RT_CHECK(vx_buffer_address(uv_buf, &arg.uv_addr));
  RT_CHECK(vx_buffer_address(w_buf, &arg.w_addr));
  RT_CHECK(vx_buffer_address(feat_buf, &arg.feat_addr));
  RT_CHECK(vx_buffer_address(y_buf, &arg.y_addr));
  RT_CHECK(vx_enqueue_write(queue, uv_buf, 0, uv.data(), uv.size() * 4, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, w_buf, 0, W.data(), W.size(), 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = 8;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));

  std::vector<uint32_t> feat(NTC_REQS * NTC_WORDS);
  std::vector<int32_t>  Y(NTC_REQS * NTC_OUTS);
  RT_CHECK(vx_enqueue_read(queue, feat.data(), feat_buf, 0, feat.size() * 4, 1, &launch_ev, nullptr));
  RT_CHECK(vx_enqueue_read(queue, Y.data(), y_buf, 0, Y.size() * 4, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;

  // (d) every sampled word against the host TEX model. Word w = stage * 4 + mip.
  std::vector<uint32_t> feat_ref(NTC_REQS * NTC_WORDS);
  for (uint32_t r = 0; r < NTC_REQS; ++r) {
    for (uint32_t w = 0; w < NTC_WORDS; ++w) {
      uint32_t s = w / kMips, mip = w % kMips;
      uint32_t ref = ref_sample(pyr[s], kLogDim[s], uv[2 * r], uv[2 * r + 1], mip);
      feat_ref[r * NTC_WORDS + w] = ref;
      uint32_t got = feat[r * NTC_WORDS + w];
      if (got != ref && errors++ < 8) {
        printf("TEX mismatch req=%u stage=%u mip=%u: got 0x%08x expected 0x%08x\n", r, s, mip, got, ref);
      }
    }
  }
  int tex_errors = errors;

  // (b)+(c) Y = X * W^T with X built from the reference features, byte 0 first.
  for (uint32_t r = 0; r < NTC_REQS; ++r) {
    for (uint32_t n = 0; n < NTC_OUTS; ++n) {
      int32_t acc = 0;
      for (uint32_t k = 0; k < NTC_FEATS; ++k) {
        uint32_t word = feat_ref[r * NTC_WORDS + k / 4];
        int32_t x = (int32_t)((word >> (8 * (k % 4))) & 0xff) - 128;
        acc += x * (int32_t)W[n * NTC_FEATS + k];
      }
      int32_t got = Y[r * NTC_OUTS + n];
      if (got != acc && errors++ < 16) {
        printf("MMA mismatch req=%u out=%u: got %d expected %d\n", r, n, got, acc);
      }
    }
  }

  cleanup();
  printf("TEX words checked: %u (%d errors); MMA outputs checked: %u (%d errors)\n",
         NTC_REQS * NTC_WORDS, tex_errors, NTC_REQS * NTC_OUTS, errors - tex_errors);
  if (errors) {
    printf("FAILED!\n");
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
