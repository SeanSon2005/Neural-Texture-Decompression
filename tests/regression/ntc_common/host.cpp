// Host for the neural texture decoders (built twice: NTC_TCU=0 -> ntc_scalar, 1 -> ntc_tcu).
//
// Reads an export written by ntc/train.py + `python -m ntc.vortex_io prepare`, uploads the
// feature atlases (one per TEX stage), weights and request vectors, runs the kernel and
// compares every decoded byte against the reference decoder's expected file.
//
//   ntc_{scalar,tcu} -e <export dir> [-v vectors] [-n max requests] [-o occupancy] [-d dump]
//
// TCU tiles are formed FIFO (the "unbatched tensor" path): a tile takes consecutive requests
// while they share a material, up to `-o` rows; a material change closes the tile early.

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <vortex2.h>
#include <VX_types.h>
#include <vx_gfx_abi.h>
#include "ntc_common.h"

#ifndef NTC_TCU
#error "define NTC_TCU=0 (scalar) or 1 (tensor)"
#endif

using namespace vortex;

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

static const char* kPath = NTC_TCU ? "tcu" : "scalar";

const char* kernel_file = "kernel.vxbin";
std::string export_dir;
std::string vec_name = "mixed";
uint32_t    max_reqs = 0;
uint32_t    occupancy = NTC_TILE_M;
std::string dump_file;

vx_device_h device  = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
std::vector<vx_buffer_h> buffers;
// vx_enqueue_write is asynchronous: the queue worker reads the host pointer later, so every
// upload source is parked here until the device is released.
std::vector<std::vector<uint8_t>> staging;

static void cleanup() {
  if (!device) return;
  for (auto b : buffers) vx_buffer_release(b);
  if (kernel)  vx_kernel_release(kernel);
  if (module_) vx_module_release(module_);
  if (queue)   vx_queue_release(queue);
  vx_device_dump_perf(device, stdout);
  vx_device_release(device);
  device = nullptr;
}

static void fail(const std::string& msg) {
  std::cout << "Error: " << msg << std::endl;
  cleanup();
  exit(-1);
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:e:v:n:o:d:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'e': export_dir = optarg; break;
    case 'v': vec_name = optarg; break;
    case 'n': max_reqs = atoi(optarg); break;
    case 'o': occupancy = atoi(optarg); break;
    case 'd': dump_file = optarg; break;
    default:
      std::cout << "Usage: -e export_dir [-v vectors=mixed] [-n max_requests] "
                   "[-o tile occupancy 1..8] [-d dump.bin] [-k kernel]" << std::endl;
      exit(c == 'h' ? 0 : -1);
    }
  }
  if (export_dir.empty()) {
    std::cout << "Error: -e <export dir> is required" << std::endl;
    exit(-1);
  }
  if (occupancy < 1 || occupancy > NTC_TILE_M) {
    std::cout << "Error: -o must be in 1.." << NTC_TILE_M << std::endl;
    exit(-1);
  }
}

static std::vector<uint8_t> read_file(const std::string& name) {
  std::string path = export_dir + "/" + name;
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------- export (vortex.txt)

struct Layer { uint32_t out, in, shift, w_off, b_off; };
struct Stage { std::string file; uint32_t log_w, log_h, bytes, mip_off[NTC_MIPS]; };
struct Export {
  uint32_t hidden = 0, materials = 0, log_tiles_x = 0, log_tiles_y = 0, local_min = 0, local_max = 0;
  Stage stage[NTC_STAGES];
  std::vector<std::vector<Layer>> layers;   // [material][layer]
};

static Export load_export() {
  Export e;
  auto bytes = read_file("vortex.txt");
  std::istringstream in(std::string(bytes.begin(), bytes.end()));
  std::string key;
  while (in >> key) {
    if (key == "hidden") in >> e.hidden;
    else if (key == "materials") { in >> e.materials; e.layers.resize(e.materials); }
    else if (key == "log_tiles") in >> e.log_tiles_x >> e.log_tiles_y;
    else if (key == "local") in >> e.local_min >> e.local_max;
    else if (key == "stage") {
      uint32_t s; in >> s;
      if (s >= NTC_STAGES) fail("bad stage in vortex.txt");
      auto& st = e.stage[s];
      in >> st.file >> st.log_w >> st.log_h >> st.bytes;
      for (auto& o : st.mip_off) in >> o;
    } else if (key == "material") {
      uint32_t m, nl; std::string name; in >> m >> name >> nl;
      if (m >= e.materials || nl != NTC_LAYERS) fail("bad material line in vortex.txt");
      e.layers[m].resize(nl);
      for (auto& l : e.layers[m]) in >> l.out >> l.in >> l.shift >> l.w_off >> l.b_off;
    } else {
      fail("unknown key '" + key + "' in vortex.txt");
    }
  }
  if (!in.eof()) fail("malformed vortex.txt");
  if (e.hidden % 8 || e.hidden == 0 || e.hidden > 64) fail("hidden must be 8..64, a multiple of 8");
  if (NTC_TCU && e.hidden != 16 && e.hidden != 32 && e.hidden != 64)
    fail("ntc_tcu is instantiated for hidden 16, 32 and 64 only");
  for (auto& ls : e.layers) {
    if (ls.size() != NTC_LAYERS || ls[0].in != NTC_FEATS || ls[0].out != e.hidden ||
        ls[1].in != e.hidden || ls[1].out != e.hidden || ls[2].in != e.hidden ||
        ls[2].out != NTC_CHANNELS)
      fail("layer shapes do not match 32 -> H -> H -> 8");
    for (auto& l : ls) if (l.shift < 1 || l.shift > 30) fail("shift out of range");
  }
  return e;
}

// ---------------------------------------------------------------- weights

// Output row p of an 8-row group holds logical neuron 8t + kPerm[p]. With this order the
// accumulator lanes of one layer hold exactly the bytes their A-fragment registers need
// for the next layer (see ntc_tcu/kernel.cpp), so no data crosses lanes between layers.
static const uint32_t kPerm[8] = {0, 4, 1, 5, 2, 6, 3, 7};

static std::vector<uint8_t> build_weights(const Export& e, const std::vector<uint8_t>& raw,
                                          std::vector<ntc_layer_t>& table) {
  table.resize(e.materials * NTC_LAYERS);
  if (!NTC_TCU) {
    for (uint32_t m = 0; m < e.materials; ++m)
      for (uint32_t l = 0; l < NTC_LAYERS; ++l) {
        auto& L = e.layers[m][l];
        table[m * NTC_LAYERS + l] = {L.w_off, L.b_off, L.shift, L.in};
      }
    return raw;
  }
  std::vector<uint8_t> blob;
  for (uint32_t m = 0; m < e.materials; ++m) {
    for (uint32_t l = 0; l < NTC_LAYERS; ++l) {
      auto& L = e.layers[m][l];
      uint32_t out_pad = (L.out + 7) / 8 * 8, in_pad = (L.in + 31) / 32 * 32;
      auto w = reinterpret_cast<const int8_t*>(raw.data() + L.w_off);
      auto b = reinterpret_cast<const int32_t*>(raw.data() + L.b_off);
      uint32_t w_off = blob.size();
      blob.resize(w_off + out_pad * in_pad, 0);
      uint32_t b_off = blob.size();
      blob.resize(b_off + out_pad * 4, 0);
      auto wt = reinterpret_cast<int8_t*>(blob.data() + w_off);
      auto bt = reinterpret_cast<int32_t*>(blob.data() + b_off);
      for (uint32_t row = 0; row < out_pad; ++row) {
        uint32_t n = (row & ~7u) + kPerm[row & 7];
        if (n >= L.out) continue;
        bt[row] = b[n];
        for (uint32_t k = 0; k < L.in; ++k) wt[row * in_pad + k] = w[n * L.in + k];
      }
      table[m * NTC_LAYERS + l] = {w_off, b_off, L.shift, in_pad};
    }
  }
  return blob;
}

// ---------------------------------------------------------------- tiles

static std::vector<ntc_tile_t> build_tiles(const std::vector<uint32_t>& req, uint32_t n, uint32_t occ) {
  std::vector<ntc_tile_t> tiles;
  for (uint32_t i = 0; i < n;) {
    ntc_tile_t t = {};
    t.material = req[3 * i];
    while (i < n && t.count < occ && req[3 * i] == t.material) t.rows[t.count++] = i++;
    for (uint32_t r = t.count; r < NTC_TILE_M; ++r) t.rows[r] = t.rows[0];
    tiles.push_back(t);
  }
  return tiles;
}

// ---------------------------------------------------------------- main

template <typename T>
static uint64_t upload(const T* data, size_t bytes, int flags = VX_MEM_READ) {
  vx_buffer_h buf;
  uint64_t addr;
  RT_CHECK(vx_buffer_create(device, std::max<size_t>(bytes, 4), flags, &buf));
  buffers.push_back(buf);
  RT_CHECK(vx_buffer_address(buf, &addr));
  if (data && bytes) {
    auto p = reinterpret_cast<const uint8_t*>(data);
    staging.emplace_back(p, p + bytes);
    RT_CHECK(vx_enqueue_write(queue, buf, 0, staging.back().data(), bytes, 0, nullptr, nullptr));
  }
  return addr;
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  Export e = load_export();

  auto req_bytes = read_file("vec_" + vec_name + "_req.bin");
  auto exp_bytes = read_file("vec_" + vec_name + "_exp.bin");
  uint32_t total = req_bytes.size() / 12;
  if (exp_bytes.size() != total * NTC_CHANNELS) fail("request/expected size mismatch");
  uint32_t n = max_reqs ? std::min(max_reqs, total) : total;
  std::vector<uint32_t> req(n * 3);
  memcpy(req.data(), req_bytes.data(), n * 12);
  for (uint32_t i = 0; i < n; ++i)
    if (req[3 * i] >= e.materials) fail("request material out of range");

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  if (!(isa_flags & VX_ISA_EXT_TEX)) fail("needs the TEX extension");
  if (NTC_TCU && !(isa_flags & VX_ISA_EXT_TCU)) fail("needs the TCU extension");
  if (num_threads != NTC_TILE_M) fail("needs NUM_THREADS=8");

  // Feature atlases: one per TEX stage, bilinear, clamp (the edge clamp keeps taps in-tile).
  for (uint32_t s = 0; s < NTC_STAGES; ++s) {
    auto& st = e.stage[s];
    staging.push_back(read_file(st.file));
    auto& texels = staging.back();
    if (texels.size() != st.bytes) fail("atlas size mismatch: " + st.file);
    vx_buffer_h buf;
    uint64_t addr;
    RT_CHECK(vx_buffer_create(device, texels.size(), VX_MEM_READ | VX_MEM_PHYS, &buf));
    buffers.push_back(buf);
    RT_CHECK(vx_buffer_address(buf, &addr));
    if (addr % 64) fail("atlas buffer not 64-byte aligned");
    RT_CHECK(vx_enqueue_write(queue, buf, 0, texels.data(), texels.size(), 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_STAGE,  s, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_LOGDIM, (st.log_h << 16) | st.log_w, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_FORMAT, VX_TEX_FORMAT_A8R8G8B8, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_WRAP,   (VX_TEX_WRAP_CLAMP << 16) | VX_TEX_WRAP_CLAMP, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_FILTER, VX_TEX_FILTER_BILINEAR, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_ADDR,   addr / 64, 0, nullptr, nullptr));
    for (uint32_t l = 0; l <= (uint32_t)VX_TEX_LOD_MAX; ++l) {
      uint32_t off = st.mip_off[l < NTC_MIPS ? l : NTC_MIPS - 1];
      RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_MIPOFF_BASE + l, off, 0, nullptr, nullptr));
    }
  }

  std::vector<ntc_layer_t> table;
  auto blob = build_weights(e, read_file("weights.bin"), table);
  auto origins = read_file("materials.bin");
  if (origins.size() != e.materials * 8) fail("materials.bin size mismatch");

  std::vector<ntc_tile_t> tiles;
  if (NTC_TCU) tiles = build_tiles(req, n, occupancy);

  kernel_arg_t arg = {};
  arg.req_addr   = upload(req.data(), req.size() * 4);
  arg.tile_addr  = upload(tiles.data(), tiles.size() * sizeof(ntc_tile_t));
  arg.mat_addr   = upload(origins.data(), origins.size());
  arg.layer_addr = upload(table.data(), table.size() * sizeof(ntc_layer_t));
  arg.wt_addr    = upload(blob.data(), blob.size());
  arg.out_addr   = upload<uint8_t>(nullptr, n * NTC_CHANNELS, VX_MEM_WRITE);
  arg.num_items  = NTC_TCU ? tiles.size() : n;
  arg.hidden     = e.hidden;
  arg.log_tiles_x = e.log_tiles_x;
  arg.log_tiles_y = e.log_tiles_y;
  arg.local_min  = e.local_min;
  arg.local_max  = e.local_max;
  vx_buffer_h out_buf = buffers.back();

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Scalar: one thread per request. TCU: one 8-thread block (= one warp) per tile.
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = NTC_TCU ? tiles.size() : (n + NTC_TILE_M - 1) / NTC_TILE_M;
  li.block_dim[0] = NTC_TILE_M;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));

  std::vector<uint8_t> out(n * NTC_CHANNELS);
  RT_CHECK(vx_enqueue_read(queue, out.data(), out_buf, 0, out.size(), 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  uint32_t errors = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const uint8_t* got = &out[i * NTC_CHANNELS];
    const uint8_t* ref = &exp_bytes[i * NTC_CHANNELS];
    if (memcmp(got, ref, NTC_CHANNELS) == 0) continue;
    if (errors++ < 8) {
      printf("mismatch req=%u material=%u:", i, req[3 * i]);
      for (int c = 0; c < NTC_CHANNELS; ++c) printf(" %u/%u", got[c], ref[c]);
      printf("  (got/expected)\n");
    }
  }

  if (!dump_file.empty()) {
    std::ofstream f(dump_file, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()), out.size());
  }

  cleanup();
  uint32_t rows = NTC_TCU ? tiles.size() * NTC_TILE_M : 0;
  printf("NTC: path=%s hidden=%u vectors=%s requests=%u tiles=%zu valid_rows=%u tile_rows=%u occupancy_cap=%u\n",
         kPath, e.hidden, vec_name.c_str(), n, tiles.size(), NTC_TCU ? n : 0, rows, occupancy);
  printf("Requests checked: %u (%u mismatches)\n", n, errors);
  if (errors) {
    printf("FAILED!\n");
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
