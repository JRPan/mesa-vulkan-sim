/*
 * Golden-image verification of vulkan-sim against llvmpipe (GPGPUSIM_VERIFY=1).
 *
 * Per draw: seed vulkan-sim's FBO/depth with llvmpipe's pre-draw state, let
 * both render the draw, then compare in the attachment's stored format.
 * Output: <dir>/verify.csv (one row per draw), <dir>/draw_<n>_{golden,sim,diff}.ppm,
 * and <dir>/verify_vs.csv (VS outputs per draw and attribute, vs llvmpipe's VS).
 * Methodology: docs/verification.md.
 */

#include "lp_gpgpusim_verify.h"
#include "lp_gpgpusim_feed.h"
#include "lp_gpgpusim_sample.h"
#include "lp_gpgpusim_blend.h"
#include "lp_state_fs.h"

#include "lp_context.h"
#include "lp_flush.h"
#include "pipe/p_context.h"
#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <immintrin.h>

#include "../../frontends/lavapipe/gpgpusim_calls_from_mesa.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_to_ptx.h"
#include "draw/draw_context.h"
#include "draw/draw_vertex_header.h"
#include "draw/draw_vs.h"
#include "pipe/p_shader_tokens.h"

#define VFY_MAX_VS_OUT 32

struct vs_attr {
   char id[64];
   unsigned slot;
   bool is_pos;
   uint64_t compared, ulp_mismatch;
   uint32_t max_ulp;
   double max_abs, sum_abs;
};

static struct {
   FILE *csv;
   unsigned n;
   struct vs_attr a[VFY_MAX_VS_OUT];
   uint64_t vertices, missing;
} vsv;

static struct {
   bool initialized, enabled;
   char dir[512];
   FILE *csv;
   unsigned draw;
   /* pre-draw state of the current draw */
   unsigned w, h;
   enum pipe_format color_format;
   uint8_t *pre_raw;   /* color, stored format, tightly packed rows */
   float *pre_rgba;    /* color unpacked to float RGBA */
   float *pre_depth;   /* NULL if no depth attachment */
} vfy;

bool
lp_gpgpusim_verify_enabled(void)
{
   if (!vfy.initialized) {
      vfy.initialized = true;
      const char *v = getenv("GPGPUSIM_VERIFY");
      vfy.enabled = v && strcmp(v, "0") != 0;
      if (lp_gpgpusim_feed_enabled() && !vfy.enabled) {
         fprintf(stderr, "gpgpusim: GPGPUSIM_FRAG_FEED requires GPGPUSIM_VERIFY=1 (it seeds depth and runs the PS)\n");
         abort();
      }
      if (vfy.enabled) {
         const char *dir = getenv("GPGPUSIM_VERIFY_DIR");
         const char *mesa_root = getenv("MESA_ROOT");
         if (dir)
            snprintf(vfy.dir, sizeof(vfy.dir), "%s", dir);
         else
            snprintf(vfy.dir, sizeof(vfy.dir), "%s/../fb/verify", mesa_root ? mesa_root : ".");
         if (mkdir(vfy.dir, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "gpgpusim verify: cannot create %s: %s\n", vfy.dir, strerror(errno));
            abort();
         }
         char path[600];
         snprintf(path, sizeof(path), "%s/verify.csv", vfy.dir);
         vfy.csv = fopen(path, "w");
         if (!vfy.csv) {
            fprintf(stderr, "gpgpusim verify: cannot open %s\n", path);
            abort();
         }
         fprintf(vfy.csv, "draw,width,height,format,golden_px,sim_px,both_px,golden_only_px,sim_only_px,"
                          "exact_pct,within1_pct,max_lsb,mean_lsb,psnr_db,"
                          "both_exact_pct,both_max_lsb,both_mean_lsb,depth_max_abs,depth_mismatch_px\n");
         fflush(vfy.csv);
         printf("gpgpusim verify: writing to %s\n", vfy.dir);
      }
   }
   return vfy.enabled;
}

/* Copies a surface into tightly packed rows; returns false if absent. */
static bool
read_surface(struct pipe_context *pipe, struct pipe_surface *surf, uint8_t **out)
{
   if (!surf || !surf->texture)
      return false;
   struct pipe_transfer *transfer;
   struct pipe_box box;
   u_box_2d_zslice(0, 0, surf->u.tex.first_layer, surf->width, surf->height, &box);
   const uint8_t *map = pipe->texture_map(pipe, surf->texture, surf->u.tex.level,
                                          PIPE_MAP_READ, &box, &transfer);
   if (!map) {
      fprintf(stderr, "gpgpusim verify: failed to map surface\n");
      abort();
   }
   unsigned row = util_format_get_stride(surf->format, surf->width);
   *out = MALLOC(row * surf->height);
   for (unsigned y = 0; y < surf->height; y++)
      memcpy(*out + y * row, map + y * transfer->stride, row);
   pipe->texture_unmap(pipe, transfer);
   return true;
}

/* The comparison works on stored bytes and so needs 4 x 8-bit normalized channels. */
static void
check_color_format(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   bool ok = desc->block.bits == 32 && desc->nr_channels == 4 &&
             desc->layout == UTIL_FORMAT_LAYOUT_PLAIN;
   for (unsigned c = 0; ok && c < 4; c++)
      ok = desc->channel[c].size == 8 && desc->channel[c].normalized &&
           desc->channel[c].type == UTIL_FORMAT_TYPE_UNSIGNED;
   if (!ok) {
      fprintf(stderr, "gpgpusim verify: color format %s is not 4x8-bit UNORM/SRGB; "
                      "extend lp_gpgpusim_verify.c to compare it\n", util_format_name(format));
      abort();
   }
}

/* Distance in representable floats; NaN vs non-NaN is maximal. */
static uint32_t
ulp_dist(float a, float b)
{
   if (isnan(a) || isnan(b))
      return isnan(a) && isnan(b) ? 0 : UINT32_MAX;
   int32_t ia, ib;
   memcpy(&ia, &a, 4);
   memcpy(&ib, &b, 4);
   int64_t oa = ia < 0 ? (int64_t)INT32_MIN - ia : ia;
   int64_t ob = ib < 0 ? (int64_t)INT32_MIN - ib : ib;
   int64_t d = oa > ob ? oa - ob : ob - oa;
   return d > UINT32_MAX ? UINT32_MAX : (uint32_t)d;
}

static void
vs_output_tap(void *data, const struct vertex_header *verts, unsigned stride,
              unsigned count, const unsigned *elts, unsigned start, unsigned instance)
{
   for (unsigned i = 0; i < count; i++) {
      const struct vertex_header *v =
         (const struct vertex_header *)((const uint8_t *)verts + (size_t)i * stride);
      unsigned index = elts ? elts[i] : start + i;
      vsv.vertices++;
      for (unsigned k = 0; k < vsv.n; k++) {
         struct vs_attr *a = &vsv.a[k];
         float s[4];
         unsigned nc = gpgpusim_getVSOutput(a->id, instance, index, s);
         if (!nc) {
            vsv.missing++;
            continue;
         }
         const float *g = a->is_pos ? v->clip_pos : v->data[a->slot];
         for (unsigned c = 0; c < nc; c++) {
            uint32_t u = ulp_dist(g[c], s[c]);
            double e = fabs((double)g[c] - (double)s[c]);
            a->compared++;
            if (u != 0 && a->ulp_mismatch < 4)
               printf("gpgpusim verify: VS %s instance %u vertex %u comp %u: llvmpipe %.9g, vulkan-sim %.9g\n",
                      a->id, instance, index, c, g[c], s[c]);
            a->ulp_mismatch += u != 0;
            a->max_ulp = MAX2(a->max_ulp, u);
            a->max_abs = isnan(e) ? INFINITY : MAX2(a->max_abs, e);
            a->sum_abs += isnan(e) ? 0 : e;
         }
      }
   }
}

/* Maps llvmpipe's VS output slots (driver_location) to vulkan-sim identifiers. */
static void
vs_verify_setup(struct llvmpipe_context *lp)
{
   memset(vsv.a, 0, sizeof(vsv.a));
   vsv.n = 0;
   vsv.vertices = vsv.missing = 0;

   struct draw_vertex_shader *vs = lp->vs;
   nir_shader *nir = vs ? vs->gpgpusim_nir : NULL;
   if (!nir)
      return;
   if (!vsv.csv) {
      char path[600];
      snprintf(path, sizeof(path), "%s/verify_vs.csv", vfy.dir);
      vsv.csv = fopen(path, "w");
      if (!vsv.csv) {
         fprintf(stderr, "gpgpusim verify: cannot open %s\n", path);
         abort();
      }
      fprintf(vsv.csv, "draw,attribute,components_compared,ulp_mismatch,max_ulp,max_abs,mean_abs,vertices,missing\n");
   }

   nir_foreach_shader_out_variable(var, nir) {
      if (glsl_count_attribute_slots(var->type, false) != 1) {
         printf("gpgpusim verify: VS output %s spans several slots; not compared\n",
                var->name ? var->name : "?");
         continue;
      }
      if (vsv.n == VFY_MAX_VS_OUT) {
         fprintf(stderr, "gpgpusim verify: more than %u VS outputs\n", VFY_MAX_VS_OUT);
         abort();
      }
      struct vs_attr *a = &vsv.a[vsv.n++];
      a->slot = var->data.driver_location;
      a->is_pos = var->data.location == VARYING_SLOT_POS;
      nir_to_ptx_var_location(nir, var, a->id, sizeof(a->id));
      /* Our slot assumption must agree with llvmpipe's own output semantics. */
      if (a->slot >= vs->info.num_outputs ||
          (vs->info.output_semantic_name[a->slot] == TGSI_SEMANTIC_POSITION) != a->is_pos) {
         fprintf(stderr, "gpgpusim verify: VS output %s: driver_location %u disagrees with llvmpipe "
                         "(num_outputs %u, semantic %u)\n", a->id, a->slot, vs->info.num_outputs,
                 a->slot < vs->info.num_outputs ? vs->info.output_semantic_name[a->slot] : ~0u);
         abort();
      }
   }
   draw_set_vs_output_tap(lp->draw, vs_output_tap, NULL);
}

static void
vs_verify_report(unsigned draw)
{
   for (unsigned k = 0; k < vsv.n; k++) {
      struct vs_attr *a = &vsv.a[k];
      fprintf(vsv.csv, "%u,%s,%" PRIu64 ",%" PRIu64 ",%u,%g,%g,%" PRIu64 ",%" PRIu64 "\n", draw, a->id,
              a->compared, a->ulp_mismatch, a->max_ulp, a->max_abs,
              a->compared ? a->sum_abs / a->compared : 0.0, vsv.vertices, vsv.missing);
      printf("gpgpusim verify: draw %u VS %s: %" PRIu64 " comps, %" PRIu64 " differ, max %u ULP, max abs %g\n",
             draw, a->id, a->compared, a->ulp_mismatch, a->max_ulp, a->max_abs);
   }
   fflush(vsv.csv);
   if (vsv.missing) {
      /* the check would silently compare nothing for those vertices */
      fprintf(stderr, "gpgpusim verify: draw %u VS: %" PRIu64 " of llvmpipe's vertex outputs have no vulkan-sim "
              "counterpart (vertex index / instance keys disagree)\n", draw, vsv.missing);
      abort();
   }
}

void
lp_gpgpusim_verify_pre_draw(struct llvmpipe_context *lp)
{
   vs_verify_setup(lp);

   struct pipe_context *pipe = &lp->pipe;
   struct pipe_surface *cbuf = lp->framebuffer.nr_cbufs ? lp->framebuffer.cbufs[0] : NULL;

   FREE(vfy.pre_raw);
   FREE(vfy.pre_rgba);
   FREE(vfy.pre_depth);
   vfy.pre_raw = NULL;
   vfy.pre_rgba = NULL;
   vfy.pre_depth = NULL;

   if (!cbuf) {
      if (lp_gpgpusim_feed_enabled()) {
         fprintf(stderr, "gpgpusim feed: draw %u has no color attachment; depth-only draws are unsupported\n", vfy.draw);
         abort();
      }
      fprintf(stderr, "gpgpusim verify: draw %u has no color attachment; depth-only draws are not compared\n", vfy.draw);
      return;
   }

   llvmpipe_finish(pipe, "gpgpusim verify pre-draw");

   vfy.w = cbuf->width;
   vfy.h = cbuf->height;
   vfy.color_format = cbuf->format;
   check_color_format(cbuf->format);
   if (vfy.w != lp->framebuffer.width || vfy.h != lp->framebuffer.height) {
      fprintf(stderr, "gpgpusim verify: color attachment %ux%u != framebuffer %ux%u\n",
              vfy.w, vfy.h, lp->framebuffer.width, lp->framebuffer.height);
      abort();
   }

   read_surface(pipe, cbuf, &vfy.pre_raw);
   unsigned row = util_format_get_stride(cbuf->format, vfy.w);
   vfy.pre_rgba = MALLOC(sizeof(float) * 4 * vfy.w * vfy.h);
   for (unsigned y = 0; y < vfy.h; y++)
      util_format_unpack_rgba(cbuf->format, vfy.pre_rgba + y * vfy.w * 4, vfy.pre_raw + y * row, vfy.w);

   uint8_t *zraw = NULL;
   if (read_surface(pipe, lp->framebuffer.zsbuf, &zraw)) {
      struct pipe_surface *zs = lp->framebuffer.zsbuf;
      if (zs->width != vfy.w || zs->height != vfy.h) {
         fprintf(stderr, "gpgpusim verify: depth %ux%u != color %ux%u\n", zs->width, zs->height, vfy.w, vfy.h);
         abort();
      }
      unsigned zrow = util_format_get_stride(zs->format, vfy.w);
      vfy.pre_depth = MALLOC(sizeof(float) * vfy.w * vfy.h);
      for (unsigned y = 0; y < vfy.h; y++)
         util_format_unpack_z_float(zs->format, vfy.pre_depth + y * vfy.w, zraw + y * zrow, vfy.w);
      FREE(zraw);
   }

   gpgpusim_loadFramebuffer(vfy.w, vfy.h, vfy.pre_rgba, vfy.pre_depth);
   /* Everything rasterized from here to the post-draw flush belongs to this draw. */
   if (lp_gpgpusim_feed_enabled()) {
      lp_gpgpusim_feed_begin(lp);
      lp_gpgpusim_sample_begin(lp);
      lp_gpgpusim_blend_begin(lp);
   }
}

static bool
depth_pass(enum pipe_compare_func func, float z, float stored)
{
   switch (func) {
   case PIPE_FUNC_NEVER:    return false;
   case PIPE_FUNC_LESS:     return z < stored;
   case PIPE_FUNC_EQUAL:    return z == stored;
   case PIPE_FUNC_LEQUAL:   return z <= stored;
   case PIPE_FUNC_GREATER:  return z > stored;
   case PIPE_FUNC_NOTEQUAL: return z != stored;
   case PIPE_FUNC_GEQUAL:   return z >= stored;
   case PIPE_FUNC_ALWAYS:   return true;
   default: unreachable("bad depth func");
   }
}

/* Depth-tests the recorded fragments in llvmpipe's order and hands the
 * survivors, with FS inputs keyed by vulkan-sim identifier, to vulkan-sim. */
static void
feed_to_sim(struct llvmpipe_context *lp, unsigned draw)
{
   const struct lp_fragment_shader *fs = lp->fs;
   const struct pipe_depth_stencil_alpha_state *dsa = lp->depth_stencil;
   nir_shader *nir = fs->gpgpusim_nir;
   const unsigned total = lp_gpgpusim_feed_count();
   const unsigned num_inputs = lp_gpgpusim_feed_num_inputs();

   if (!nir) {
      fprintf(stderr, "gpgpusim feed: no NIR stashed for the fragment shader\n");
      abort();
   }
   if (dsa->stencil[0].enabled || dsa->stencil[1].enabled) {
      fprintf(stderr, "gpgpusim feed: stencil test is not supported\n");
      abort();
   }

   /* FS inputs -> vulkan-sim identifiers (slot = driver_location, as in llvmpipe's setup). */
   const char *ids[PIPE_MAX_SHADER_INPUTS];
   char id_store[PIPE_MAX_SHADER_INPUTS][64];
   unsigned ncomp[PIPE_MAX_SHADER_INPUTS], slot[PIPE_MAX_SHADER_INPUTS], frac[PIPE_MAX_SHADER_INPUTS];
   unsigned nvars = 0;
   nir_foreach_shader_in_variable(var, nir) {
      unsigned s = var->data.driver_location;
      if (glsl_count_attribute_slots(var->type, false) != 1 || s >= num_inputs) {
         fprintf(stderr, "gpgpusim feed: FS input %s (driver_location %u, %u inputs) is unsupported\n",
                 var->name ? var->name : "?", s, num_inputs);
         abort();
      }
      bool flat = var->data.interpolation == INTERP_MODE_FLAT;
      if (flat != (fs->inputs[s].interp == LP_INTERP_CONSTANT) && var->data.location != VARYING_SLOT_POS) {
         fprintf(stderr, "gpgpusim feed: FS input %s: NIR flat=%d but llvmpipe interp=%u\n",
                 var->name ? var->name : "?", flat, fs->inputs[s].interp);
         abort();
      }
      nir_to_ptx_var_location(nir, var, id_store[nvars], sizeof(id_store[0]));
      ids[nvars] = id_store[nvars];
      ncomp[nvars] = glsl_get_components(glsl_without_array(var->type));
      slot[nvars] = s;
      frac[nvars] = var->data.location_frac;
      if (frac[nvars] + ncomp[nvars] > 4) {
         fprintf(stderr, "gpgpusim feed: FS input %s spans more than one vec4\n", ids[nvars]);
         abort();
      }
      nvars++;
   }

   const bool dtest = dsa->depth_enabled && lp->framebuffer.zsbuf && vfy.pre_depth;
   const bool dwrite = dtest && dsa->depth_writemask;
   /* Depth is resolved here, before vulkan-sim's PS runs: a fragment the PS
    * discards would still have written depth, and PS-written depth is ignored. */
   if (dwrite && (nir->info.fs.uses_discard || nir->info.fs.uses_demote)) {
      fprintf(stderr, "gpgpusim feed: draw %u: FS can discard with depth writes on; late depth is not modelled\n", draw);
      abort();
   }
   if (dtest && (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH))) {
      fprintf(stderr, "gpgpusim feed: draw %u: FS writes depth; not modelled\n", draw);
      abort();
   }
   const enum pipe_format zf = dtest ? lp->framebuffer.zsbuf->format : PIPE_FORMAT_NONE;
   float *zbuf = NULL;
   if (dtest) {
      zbuf = MALLOC(sizeof(float) * vfy.w * vfy.h);
      memcpy(zbuf, vfy.pre_depth, sizeof(float) * vfy.w * vfy.h);
   }

   unsigned *pix = MALLOC(sizeof(unsigned) * MAX2(total, 1));
   float *zs = MALLOC(sizeof(float) * MAX2(total, 1));
   uint8_t *helper = MALLOC(MAX2(total, 1));
   float *vals[PIPE_MAX_SHADER_INPUTS];
   for (unsigned a = 0; a < nvars; a++)
      vals[a] = MALLOC(sizeof(float) * ncomp[a] * MAX2(total, 1));
   uint8_t *hits = CALLOC(vfy.w * vfy.h, 1);
   unsigned kept = 0, overdraw = 0;

   if (total % 4) {
      fprintf(stderr, "gpgpusim feed: %u records is not whole quads\n", total);
      abort();
   }
   unsigned live = 0, helpers = 0;
   for (unsigned q = 0; q < total; q += 4) {
      /* Depth-test the quad's covered pixels in order; failures become helpers,
       * a quad with no survivor is dropped (it would write nothing). */
      unsigned p[4];
      float z[4];
      bool h[4];
      const float *in[4];
      bool any = false;
      for (unsigned l = 0; l < 4; l++) {
         lp_gpgpusim_feed_get(q + l, &p[l], &z[l], &h[l], &in[l]);
         if (h[l] || !dtest)
            continue;
         /* compare at the depth buffer's precision, as llvmpipe does */
         uint32_t packed[2] = {0, 0};
         util_format_pack_z_float(zf, packed, &z[l], 1);
         util_format_unpack_z_float(zf, &z[l], packed, 1);
         if (!depth_pass(dsa->depth_func, z[l], zbuf[p[l]]))
            h[l] = true;
         else if (dwrite)
            zbuf[p[l]] = z[l];
      }
      for (unsigned l = 0; l < 4; l++)
         any |= !h[l];
      if (!any)
         continue;
      for (unsigned l = 0; l < 4; l++) {
         if (!h[l]) {
            overdraw += hits[p[l]] == 1;
            hits[p[l]] = MIN2(hits[p[l]] + 1, 2);
            live++;
         } else {
            helpers++;
         }
         pix[kept] = h[l] ? UINT32_MAX : p[l];
         zs[kept] = z[l];
         helper[kept] = h[l];
         for (unsigned a = 0; a < nvars; a++)
            memcpy(vals[a] + (size_t)kept * ncomp[a], in[l] + slot[a] * 4 + frac[a], sizeof(float) * ncomp[a]);
         kept++;
      }
   }
   printf("gpgpusim feed: draw %u: %u quad lanes from llvmpipe, %u live + %u helper lanes after depth, "
          "%u pixels shaded more than once\n", draw, total, live, helpers, overdraw);

   gpgpusim_runFragments(kept, pix, zs, helper, dwrite, nvars, ids, ncomp, (const float *const *)vals);

   for (unsigned a = 0; a < nvars; a++)
      FREE(vals[a]);
   FREE(hits);
   FREE(helper);
   FREE(pix);
   FREE(zs);
   FREE(zbuf);
}

/* llvmpipe's linear->sRGB8 encode (lp_build_linear_to_srgb, rational-polynomial
 * path). rsqrtss gives the same approximation as the JIT's rsqrtps on this CPU. */
static float
fast_rsqrt(float x)
{
   return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
}

static uint8_t
lp_linear_to_srgb8(float x)
{
   x = x > 1.0f ? 1.0f : x;
   x = x > 0.0f ? x : 0.0f; /* also maps NaN to 0, as maxps does */
   const float a = 0.675f * 1.0622 * 255.0f;
   const float b = 0.325f * 1.0622 * 255.0f;
   const float c = -0.0620f * 255.0f;
   float x05 = x * fast_rsqrt(x);
   float x0375 = fast_rsqrt(fast_rsqrt(x05 * x));
   float pow_final = a * x0375 + fmaf(b, x05, c);
   float lin = x * (12.92f * 255.0f);
   float r = x <= 0.0031308f ? lin : pow_final;
   return (uint8_t)CLAMP(lrintf(r), 0, 255);
}

/* Packs a row of float RGBA as llvmpipe's FS output stage would; sRGB only
 * differs from util_format (approximate encode), UNORM uses util_format. */
void
lp_gpgpusim_pack_like_llvmpipe(enum pipe_format fmt, uint8_t *dst, const float *src, unsigned w)
{
   const struct util_format_description *desc = util_format_description(fmt);
   if (desc->colorspace != UTIL_FORMAT_COLORSPACE_SRGB) {
      util_format_pack_rgba(fmt, dst, src, w);
      return;
   }
   for (unsigned x = 0; x < w; x++) {
      uint8_t *px = dst + x * 4;
      memset(px, 0, 4);
      for (unsigned i = 0; i < 4; i++) {
         unsigned sw = desc->swizzle[i];
         if (sw > PIPE_SWIZZLE_W)
            continue;
         float v = src[x * 4 + i];
         uint8_t out;
         if (i < 3) {
            out = lp_linear_to_srgb8(v);
         } else {
            v = v > 1.0f ? 1.0f : (v > 0.0f ? v : 0.0f);
            out = (uint8_t)lrintf(v * 255.0f);
         }
         px[desc->channel[sw].shift / 8] = out;
      }
   }
}

/* Display RGB of a stored 4x8 pixel (stored, i.e. sRGB-encoded, values). */
static void
display_rgb(const struct util_format_description *desc, const uint8_t *px, uint8_t rgb[3])
{
   for (unsigned i = 0; i < 3; i++) {
      unsigned sw = desc->swizzle[i];
      rgb[i] = sw < 4 ? px[desc->channel[sw].shift / 8] : (sw == PIPE_SWIZZLE_1 ? 255 : 0);
   }
}

static void
write_ppm(const char *name, unsigned w, unsigned h, const uint8_t *rgb)
{
   char path[700];
   snprintf(path, sizeof(path), "%s/%s", vfy.dir, name);
   FILE *fp = fopen(path, "wb");
   if (!fp) {
      fprintf(stderr, "gpgpusim verify: cannot write %s\n", path);
      abort();
   }
   fprintf(fp, "P6\n%u %u\n255\n", w, h);
   fwrite(rgb, 3, (size_t)w * h, fp);
   fclose(fp);
}

void
lp_gpgpusim_verify_post_draw(struct llvmpipe_context *lp)
{
   unsigned draw = vfy.draw++;
   struct pipe_context *pipe = &lp->pipe;
   draw_set_vs_output_tap(lp->draw, NULL, NULL);
   llvmpipe_finish(pipe, "gpgpusim verify post-draw");
   if (lp_gpgpusim_feed_enabled()) {
      lp_gpgpusim_feed_end();
      if (gpgpusim_fragmentsPending())
         feed_to_sim(lp, draw);
   }
   if (!gpgpusim_lastDrawRendered()) {
      printf("gpgpusim verify: draw %u skipped by vulkan-sim, not compared\n", draw);
      return;
   }
   vs_verify_report(draw);
   if (!vfy.pre_raw)
      return;

   const unsigned w = vfy.w, h = vfy.h, n = w * h;
   const enum pipe_format fmt = vfy.color_format;
   const struct util_format_description *desc = util_format_description(fmt);
   const unsigned row = util_format_get_stride(fmt, w);

   uint8_t *golden = NULL;
   read_surface(pipe, lp->framebuffer.cbufs[0], &golden);

   unsigned sw, sh;
   const float *sim_rgba, *sim_depth;
   gpgpusim_getFramebuffer(&sw, &sh, &sim_rgba, &sim_depth);
   if (sw != w || sh != h) {
      fprintf(stderr, "gpgpusim verify: vulkan-sim FBO %ux%u != golden %ux%u\n", sw, sh, w, h);
      abort();
   }
   uint8_t *sim = MALLOC(row * h);
   for (unsigned y = 0; y < h; y++)
      lp_gpgpusim_pack_like_llvmpipe(fmt, sim + y * row, sim_rgba + y * w * 4, w);

   unsigned golden_px = 0, sim_px = 0, both_px = 0, golden_only = 0, sim_only = 0;
   unsigned exact = 0, within1 = 0, max_lsb = 0, both_exact = 0, both_max = 0;
   double sum_lsb = 0, sum_sq = 0, both_sum = 0;
   uint8_t *img_g = MALLOC(3 * n), *img_s = MALLOC(3 * n), *img_d = MALLOC(3 * n);

   for (unsigned p = 0; p < n; p++) {
      const uint8_t *g = golden + p * 4, *s = sim + p * 4, *pre = vfy.pre_raw + p * 4;
      bool g_wrote = memcmp(g, pre, 4) != 0;
      bool s_wrote = memcmp(s, pre, 4) != 0;
      unsigned d = 0;
      for (unsigned c = 0; c < 4; c++) {
         unsigned e = abs((int)g[c] - (int)s[c]);
         d = MAX2(d, e);
         sum_sq += (double)e * e;
      }
      golden_px += g_wrote;
      sim_px += s_wrote;
      exact += d == 0;
      within1 += d <= 1;
      max_lsb = MAX2(max_lsb, d);
      sum_lsb += d;
      if (g_wrote && s_wrote) {
         both_px++;
         both_exact += d == 0;
         both_max = MAX2(both_max, d);
         both_sum += d;
      }
      golden_only += g_wrote && !s_wrote;
      sim_only += s_wrote && !g_wrote;

      display_rgb(desc, g, img_g + p * 3);
      display_rgb(desc, s, img_s + p * 3);
      /* diff: red = golden-only coverage, blue = sim-only, gray = value error x8 */
      uint8_t *o = img_d + p * 3;
      if (g_wrote && !s_wrote) {
         o[0] = 255; o[1] = 0; o[2] = 0;
      } else if (s_wrote && !g_wrote) {
         o[0] = 0; o[1] = 0; o[2] = 255;
      } else {
         o[0] = o[1] = o[2] = MIN2(d * 8, 255);
      }
   }

   float depth_max = 0;
   unsigned depth_bad = 0;
   uint8_t *zraw = NULL;
   if (vfy.pre_depth && read_surface(pipe, lp->framebuffer.zsbuf, &zraw)) {
      enum pipe_format zf = lp->framebuffer.zsbuf->format;
      unsigned zrow = util_format_get_stride(zf, w);
      float *zg = MALLOC(sizeof(float) * w);
      for (unsigned y = 0; y < h; y++) {
         util_format_unpack_z_float(zf, zg, zraw + y * zrow, w);
         for (unsigned x = 0; x < w; x++) {
            float e = fabsf(zg[x] - sim_depth[y * w + x]);
            depth_max = MAX2(depth_max, e);
            depth_bad += e > 1e-6f;
         }
      }
      FREE(zg);
      FREE(zraw);
   }

   double mse = sum_sq / (4.0 * n);
   double psnr = mse == 0 ? INFINITY : 10.0 * log10(255.0 * 255.0 / mse);
   fprintf(vfy.csv, "%u,%u,%u,%s,%u,%u,%u,%u,%u,%.3f,%.3f,%u,%.4f,%.2f,%.3f,%u,%.4f,%g,%u\n",
           draw, w, h, util_format_short_name(fmt), golden_px, sim_px, both_px, golden_only, sim_only,
           100.0 * exact / n, 100.0 * within1 / n, max_lsb, sum_lsb / n, psnr,
           both_px ? 100.0 * both_exact / both_px : 100.0, both_max, both_px ? both_sum / both_px : 0.0,
           depth_max, depth_bad);
   fflush(vfy.csv);
   printf("gpgpusim verify: draw %u: exact %.2f%%, max %u LSB, PSNR %.2f dB, coverage golden %u / sim %u "
          "(golden-only %u, sim-only %u), depth max err %g\n",
          draw, 100.0 * exact / n, max_lsb, psnr, golden_px, sim_px, golden_only, sim_only, depth_max);

   char name[64];
   snprintf(name, sizeof(name), "draw_%u_golden.ppm", draw);
   write_ppm(name, w, h, img_g);
   snprintf(name, sizeof(name), "draw_%u_sim.ppm", draw);
   write_ppm(name, w, h, img_s);
   snprintf(name, sizeof(name), "draw_%u_diff.ppm", draw);
   write_ppm(name, w, h, img_d);

   FREE(img_g);
   FREE(img_s);
   FREE(img_d);
   FREE(sim);
   FREE(golden);
}
