// Samples a texture level with Vortex's own host TEX model (gfx_frag_tex.h), so
// the Python port in ntc/tex_model.py can be checked against it.
// stdin : u32 log_w, log_h, filter, wrap, n; then 2^(log_w+log_h) texel words; then n (u, v) pairs.
// stdout: n texel words.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <gfx_frag_tex.h>

static bool rd(void* p, size_t n) { return fread(p, 1, n, stdin) == n; }

int main() {
  uint32_t hdr[5];
  if (!rd(hdr, sizeof(hdr))) return 1;
  uint32_t log_w = hdr[0], log_h = hdr[1], filter = hdr[2], wrap = hdr[3], n = hdr[4];
  std::vector<uint32_t> texels(1u << (log_w + log_h));
  std::vector<uint32_t> uv(2 * n), out(n);
  if (!rd(texels.data(), texels.size() * 4) || !rd(uv.data(), uv.size() * 4)) return 1;
  for (uint32_t i = 0; i < n; ++i) {
    auto req = gfx_tex::tex_compute_request(0, (log_h << 16) | log_w, VX_TEX_FORMAT_A8R8G8B8, filter,
                                            (wrap << 16) | wrap, (int32_t)uv[2 * i], (int32_t)uv[2 * i + 1], 0);
    uint32_t t[4] = {};
    for (int k = 0; k < (filter == VX_TEX_FILTER_BILINEAR ? 4 : 1); ++k) t[k] = texels[req.addr[k] / 4];
    out[i] = gfx_tex::tex_apply_filter(req, t);
  }
  fwrite(out.data(), 4, n, stdout);
  return 0;
}
