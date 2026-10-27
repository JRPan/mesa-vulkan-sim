/*
 * llvmpipe's own texture sampling, callable from vulkan-sim (fragment feed).
 *
 * For each (FS variant, texture, sampler, LOD mode) we JIT a small function
 * with the same lp_build_sample_soa code and static sampler key the FS JIT
 * uses, running on the live jit_resources of the current draw. vulkan-sim's
 * tex/txl hand it one 2x2 quad (lanes TL, TR, BL, BR), so implicit LOD comes
 * from the same quad derivatives. See docs/verification.md.
 */

#include "lp_gpgpusim_sample.h"
#include "lp_gpgpusim_feed.h"

#include "lp_context.h"
#include "lp_jit.h"
#include "lp_rast.h"
#include "lp_setup_context.h"
#include "lp_state_fs.h"
#include "lp_tex_sample.h"
#include "gallivm/lp_bld_format.h"
#include "gallivm/lp_bld_init.h"
#include "gallivm/lp_bld_debug.h"
#include "gallivm/lp_bld_sample.h"
#include "gallivm/lp_bld_type.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../frontends/lavapipe/gpgpusim_calls_from_mesa.h"

typedef void (*lp_sample_fn)(const struct lp_jit_resources *resources,
                             struct lp_jit_thread_data *thread_data,
                             const float *s, const float *t, const float *r,
                             const float *lod, float *texel, int32_t *mip);

struct sample_entry {
   const struct llvmpipe_context *lp;
   /* shader->no is unique per FS (global counter), variant->no per variant of it;
    * pointers get reused after a variant is freed */
   unsigned shader_no, variant_no;
   unsigned texture_index, sampler_index;
   unsigned lod_mode;
   bool mip_only; /* timing variant: exports the mip decision instead of texels */
   struct gallivm_state *gallivm;
   lp_sample_fn fn;
};

static struct {
   struct llvmpipe_context *lp;
   struct util_dynarray cache; /* struct sample_entry */
   struct lp_jit_thread_data thread_data;
   unsigned length;            /* lanes per vector, as in the FS JIT */
   bool initialized;
   /* last quad sampled (all 4 lanes of a quad ask for the same thing) */
   bool last_valid;
   struct {
      unsigned texture_index, sampler_index, lod_mode, texel_bytes;
      float s[4], t[4], r[4], lod[4], texel[4][4];
      uint64_t addr[4][GPGPUSIM_MAX_TEXEL_ADDRS];
      unsigned naddr[4];
   } last;
} smp;

static lp_sample_fn
build_sample_fn(const struct lp_fragment_shader_variant *variant, unsigned tex,
                unsigned samp, unsigned lod_mode, bool mip_only, struct gallivm_state **out_gallivm)
{
   struct gallivm_state *gallivm = gallivm_create("gpgpusim_sample", smp.lp->context, NULL);
   LLVMContextRef lc = gallivm->context;
   LLVMBuilderRef b = gallivm->builder;

   /* borrow the variant type setup for jit_resources / thread_data */
   struct lp_fragment_shader_variant types;
   memset(&types, 0, sizeof(types));
   types.gallivm = gallivm;
   lp_jit_init_types(&types);

   struct lp_type type;
   memset(&type, 0, sizeof(type));
   type.floating = true;
   type.sign = true;
   type.width = 32;
   type.length = smp.length; /* same vector width as generate_fragment's fs_type */
   LLVMTypeRef vec = lp_build_vec_type(gallivm, type);
   LLVMTypeRef fptr = LLVMPointerType(LLVMFloatTypeInContext(lc), 0);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(lc);
   LLVMTypeRef args[8] = { types.jit_resources_ptr_type, types.jit_thread_data_ptr_type,
                           fptr, fptr, fptr, fptr, fptr, LLVMPointerType(i32, 0) };
   LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidTypeInContext(lc), args, 8, 0);
   LLVMValueRef func = LLVMAddFunction(gallivm->module, "gpgpusim_sample", fn_type);
   LLVMSetFunctionCallConv(func, LLVMCCallConv);
   LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(lc, func, "entry"));

   LLVMTypeRef vec_ptr = LLVMPointerType(vec, 0);
   LLVMValueRef in[4]; /* s, t, layer, lod */
   for (unsigned i = 0; i < 4; i++) {
      LLVMValueRef p = LLVMBuildBitCast(b, LLVMGetParam(func, 2 + i), vec_ptr, "");
      in[i] = LLVMBuildLoad2(b, vec, p, "");
      LLVMSetAlignment(in[i], 4);
   }

   LLVMValueRef coords[5] = { in[0], in[1], (lod_mode & (8 | 16)) ? in[2] : LLVMGetUndef(vec),
                              LLVMGetUndef(vec), LLVMGetUndef(vec) };
   LLVMValueRef offsets[3] = { NULL, NULL, NULL };
   LLVMValueRef texel[4], mip[4];

   /* sample_key as lp_build_nir_sample_key builds it for a FS tex/txb/txl:
    * lod_mode = 0 implicit | 1 bias | 2 explicit, +4 if the lod source is uniform */
   const bool has_lod = (lod_mode & 3) != 0;
   uint32_t sample_key = LP_SAMPLER_LOD_SCALAR << LP_SAMPLER_LOD_PROPERTY_SHIFT;
   if (has_lod) {
      sample_key = ((lod_mode & 3) == 1 ? LP_SAMPLER_LOD_BIAS : LP_SAMPLER_LOD_EXPLICIT)
                      << LP_SAMPLER_LOD_CONTROL_SHIFT;
      sample_key |= ((lod_mode & 4) ? LP_SAMPLER_LOD_SCALAR : LP_SAMPLER_LOD_PER_QUAD)
                       << LP_SAMPLER_LOD_PROPERTY_SHIFT;
   }

   struct lp_sampler_params params;
   memset(&params, 0, sizeof(params));
   params.type = type;
   params.sample_key = sample_key;
   params.texture_index = tex;
   params.sampler_index = samp;
   params.resources_type = types.jit_resources_type;
   params.resources_ptr = LLVMGetParam(func, 0);
   params.thread_data_type = types.jit_thread_data_type;
   params.thread_data_ptr = LLVMGetParam(func, 1);
   params.coords = coords;
   params.offsets = offsets;
   params.texel = texel;
   params.lod = has_lod ? in[3] : NULL;
   params.aniso_filter_table = lp_jit_resources_aniso_filter_table(gallivm, params.resources_type,
                                                                   params.resources_ptr);
   /* separate function so the texel path stays exactly the FS JIT's */
   params.mip_out = mip_only ? mip : NULL;

   struct lp_build_sampler_soa *sampler =
      lp_llvm_sampler_soa_create(lp_fs_variant_key_samplers(&variant->key),
                                 MAX2(variant->key.nr_samplers, variant->key.nr_sampler_views));
   sampler->emit_tex_sample(sampler, gallivm, &params);
   lp_bld_llvm_sampler_soa_destroy(sampler);

   if (mip_only) {
      for (unsigned i = 0; i < 4; i++) {
         LLVMValueRef idx = LLVMConstInt(i32, i, 0);
         LLVMBuildStore(b, mip[i], LLVMBuildGEP2(b, i32, LLVMGetParam(func, 7), &idx, 1, ""));
      }
   } else {
      LLVMValueRef out = LLVMBuildBitCast(b, LLVMGetParam(func, 6), vec_ptr, "");
      for (unsigned c = 0; c < 4; c++) {
         LLVMValueRef idx = LLVMConstInt(i32, c, 0);
         LLVMValueRef dst = LLVMBuildGEP2(b, vec, out, &idx, 1, "");
         LLVMValueRef st = LLVMBuildStore(b, texel[c], dst);
         LLVMSetAlignment(st, 4);
      }
   }
   LLVMBuildRetVoid(b);

   gallivm_verify_function(gallivm, func);
   gallivm_compile_module(gallivm);
   lp_sample_fn fn = (lp_sample_fn)gallivm_jit_function(gallivm, func);
   gallivm_free_ir(gallivm);
   *out_gallivm = gallivm;
   return fn;
}

static lp_sample_fn
get_fn(const struct lp_fragment_shader_variant *variant, unsigned tex, unsigned samp,
       unsigned lod_mode, bool mip_only)
{
   util_dynarray_foreach(&smp.cache, struct sample_entry, e) {
      if (e->shader_no == variant->shader->no && e->variant_no == variant->no && e->texture_index == tex &&
          e->sampler_index == samp && e->lod_mode == lod_mode && e->mip_only == mip_only)
         return e->fn;
   }
   struct sample_entry e = { smp.lp, variant->shader->no, variant->no, tex, samp, lod_mode, mip_only, NULL, NULL };
   e.fn = build_sample_fn(variant, tex, samp, lod_mode, mip_only, &e.gallivm);
   util_dynarray_append(&smp.cache, struct sample_entry, e);
   return e.fn;
}

static int
wrap_texel(int i, int size, unsigned wrap)
{
   switch (wrap) {
   case PIPE_TEX_WRAP_REPEAT:
      return ((i % size) + size) % size;
   case PIPE_TEX_WRAP_MIRROR_REPEAT: {
      int m = ((i % (2 * size)) + 2 * size) % (2 * size);
      return m < size ? m : 2 * size - 1 - m;
   }
   default: /* clamp variants; border texels are not fetched from memory */
      return CLAMP(i, 0, size - 1);
   }
}

/* Texel addresses one lane reads, from llvmpipe's mip decision (lane 0 of the
 * quad; llvmpipe's LOD is per quad or scalar here) and the sampler state. */
static unsigned
lane_footprint(const struct lp_jit_texture *jt, const struct lp_static_texture_state *ts,
               const struct lp_static_sampler_state *ss, const int32_t mip[4],
               float s, float t, float r, unsigned lod_mode, uint64_t *addr)
{
   const bool array = lod_mode & 8;
   unsigned wrap_s = ss->wrap_s, wrap_t = ss->wrap_t;
   int face = -1;
   if (lod_mode & 16) {
      /* cube: major axis picks the face, the other two give its 2D coords */
      const float ax = fabsf(s), ay = fabsf(t), az = fabsf(r);
      float sc, tc, ma;
      if (ax >= ay && ax >= az) {
         face = s > 0 ? 0 : 1; sc = s > 0 ? -r : r; tc = -t; ma = ax;
      } else if (ay >= az) {
         face = t > 0 ? 2 : 3; sc = s; tc = t > 0 ? r : -r; ma = ay;
      } else {
         face = r > 0 ? 4 : 5; sc = r > 0 ? s : -s; tc = -t; ma = az;
      }
      ma = ma > 0 ? ma : 1.0f;
      s = 0.5f * (sc / ma + 1.0f);
      t = 0.5f * (tc / ma + 1.0f);
      wrap_s = wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   }
   /* lod_positive is not computed when min == mag filter (mip[2] == 2) */
   const unsigned filter = mip[2] == 1 || mip[2] == -1 ? ss->min_img_filter : ss->mag_img_filter;
   float lod_fpart;
   memcpy(&lod_fpart, &mip[3], 4);
   int levels[2] = { mip[0] >= 0 ? mip[0] : (int)jt->first_level, -1 };
   /* llvmpipe only fetches the second level when it has weight */
   if (ss->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR && mip[1] >= 0 && mip[1] != levels[0] &&
       lod_fpart > 0.0f)
      levels[1] = mip[1];

   const struct util_format_description *desc = util_format_description(ts->format);
   const unsigned bw = desc->block.width, bh = desc->block.height, bytes = desc->block.bits / 8;
   /* array layer: round to nearest, clamp (as lp_build_layer_coord) */
   const uint64_t layer = face >= 0 ? (uint64_t)face :
                          array ? (uint64_t)CLAMP((int)lrintf(r), 0, (int)jt->depth - 1) : 0;
   unsigned n = 0;
   for (unsigned li = 0; li < 2 && levels[li] >= 0; li++) {
      const int L = levels[li];
      const int w = MAX2(1, (int)jt->width >> L), h = MAX2(1, (int)jt->height >> L);
      const float u = ss->normalized_coords ? s * w : s, v = ss->normalized_coords ? t * h : t;
      int x0, y0, nx, ny;
      if (filter == PIPE_TEX_FILTER_LINEAR) {
         x0 = (int)floorf(u - 0.5f);
         y0 = (int)floorf(v - 0.5f);
         nx = ny = 2;
      } else {
         x0 = (int)floorf(u);
         y0 = (int)floorf(v);
         nx = ny = 1;
      }
      for (int dy = 0; dy < ny; dy++) {
         for (int dx = 0; dx < nx; dx++) {
            const int x = wrap_texel(x0 + dx, w, wrap_s), y = wrap_texel(y0 + dy, h, wrap_t);
            uint64_t a = (uint64_t)(uintptr_t)jt->base + jt->mip_offsets[L] + layer * jt->img_stride[L] +
                         (uint64_t)(y / bh) * jt->row_stride[L] + (uint64_t)(x / bw) * bytes;
            bool dup = false;
            for (unsigned k = 0; k < n; k++)
               dup |= addr[k] == a;
            if (!dup)
               addr[n++] = a;
         }
      }
   }
   return n;
}

/* Callback for vulkan-sim: samples one quad; texel[lane][chan]. Also returns the
 * texel addresses each lane reads (for the timing model) and their size. */
static void
sample_quad(unsigned texture_index, unsigned sampler_index, unsigned lod_mode,
            const float s[4], const float t[4], const float r[4], const float lod[4], float texel[4][4],
            uint64_t addr[4][GPGPUSIM_MAX_TEXEL_ADDRS], unsigned naddr[4], unsigned *texel_bytes)
{
   const struct lp_rast_state *state = lp_gpgpusim_feed_state();
   const struct lp_fragment_shader_variant *variant = state ? state->variant : NULL;
   if (!variant) {
      fprintf(stderr, "gpgpusim sample: no rasterizer state captured for this draw\n");
      abort();
   }
   const unsigned nunits = MAX2(variant->key.nr_samplers, variant->key.nr_sampler_views);
   if (texture_index >= nunits || sampler_index >= nunits) {
      fprintf(stderr, "gpgpusim sample: texture %u / sampler %u out of range (variant has %u)\n",
              texture_index, sampler_index, nunits);
      abort();
   }

   /* The 4 lanes of a quad call with identical inputs: reuse the last result. */
   if (smp.last_valid && smp.last.texture_index == texture_index &&
       smp.last.sampler_index == sampler_index && smp.last.lod_mode == lod_mode &&
       !memcmp(smp.last.s, s, sizeof(smp.last.s)) && !memcmp(smp.last.t, t, sizeof(smp.last.t)) &&
       !memcmp(smp.last.r, r, sizeof(smp.last.r)) && !memcmp(smp.last.lod, lod, sizeof(smp.last.lod))) {
      memcpy(texel, smp.last.texel, sizeof(smp.last.texel));
      memcpy(addr, smp.last.addr, sizeof(smp.last.addr));
      memcpy(naddr, smp.last.naddr, sizeof(smp.last.naddr));
      *texel_bytes = smp.last.texel_bytes;
      return;
   }

   /* Fill every quad of the vector with this quad; lanes 0..3 hold the result. */
   float vs[16], vt[16], vr[16], vl[16], out[4 * 16];
   int32_t mip[4];
   for (unsigned i = 0; i < smp.length; i++) {
      vs[i] = s[i % 4];
      vt[i] = t[i % 4];
      vr[i] = (lod_mode & (8 | 16)) ? r[i % 4] : 0.0f;
      vl[i] = (lod_mode & 3) ? lod[i % 4] : 0.0f;
   }
   /* the resources llvmpipe actually shaded this draw with (feed snapshot) */
   const struct lp_jit_resources *res = &state->jit_resources;
   {
      /* the JIT reads these blindly; check what it is about to sample */
      const struct lp_jit_texture *jt = &res->textures[texture_index];
      const struct lp_static_texture_state *ts =
         &lp_fs_variant_key_samplers(&variant->key)[texture_index].texture_state;
      const enum pipe_texture_target want = (lod_mode & 8) ? PIPE_TEXTURE_2D_ARRAY :
                                            (lod_mode & 16) ? PIPE_TEXTURE_CUBE : PIPE_TEXTURE_2D;
      if (!jt->base || !jt->width || !jt->height || ts->target != want) {
         fprintf(stderr, "gpgpusim sample: texture %u: base %p %ux%u, target %u, format %s "
                 "(only bound 2D textures are supported)\n", texture_index, jt->base, jt->width,
                 jt->height, ts->target, util_format_name(ts->format));
         abort();
      }
   }
   get_fn(variant, texture_index, sampler_index, lod_mode, false)(res, &smp.thread_data, vs, vt, vr, vl, out, NULL);
   get_fn(variant, texture_index, sampler_index, lod_mode, true)(res, &smp.thread_data, vs, vt, vr, vl, out, mip);
   for (unsigned l = 0; l < 4; l++)
      for (unsigned c = 0; c < 4; c++)
         texel[l][c] = out[c * smp.length + l];


   const struct lp_sampler_static_state *key = lp_fs_variant_key_samplers(&variant->key);
   const struct lp_static_texture_state *ts = &key[texture_index].texture_state;
   const struct lp_static_sampler_state *ss = &key[sampler_index].sampler_state;
   *texel_bytes = MAX2(util_format_get_blocksize(ts->format), 1);
   for (unsigned l = 0; l < 4; l++)
      naddr[l] = lane_footprint(&res->textures[texture_index], ts, ss, mip, s[l], t[l], r[l],
                                lod_mode, addr[l]);

   smp.last.texture_index = texture_index;
   smp.last.sampler_index = sampler_index;
   smp.last.lod_mode = lod_mode;
   memcpy(smp.last.s, s, sizeof(smp.last.s));
   memcpy(smp.last.t, t, sizeof(smp.last.t));
   memcpy(smp.last.r, r, sizeof(smp.last.r));
   memcpy(smp.last.lod, lod, sizeof(smp.last.lod));
   memcpy(smp.last.texel, texel, sizeof(smp.last.texel));
   memcpy(smp.last.addr, addr, sizeof(smp.last.addr));
   memcpy(smp.last.naddr, naddr, sizeof(smp.last.naddr));
   smp.last.texel_bytes = *texel_bytes;
   smp.last_valid = true;
}

void
lp_gpgpusim_sample_begin(struct llvmpipe_context *lp)
{
   if (!smp.initialized) {
      smp.initialized = true;
      util_dynarray_init(&smp.cache, NULL);
      smp.thread_data.cache = align_malloc(sizeof(struct lp_build_format_cache), 16);
      memset(smp.thread_data.cache, 0, sizeof(struct lp_build_format_cache));
      smp.length = MIN2(lp_native_vector_width / 32, 16);
      if (smp.length < 4 || smp.length % 4) {
         fprintf(stderr, "gpgpusim sample: unexpected vector length %u\n", smp.length);
         abort();
      }
   }
   /* The sample functions copy the FS JIT's per-quad LOD choice; with per-pixel
    * LOD (GALLIVM_PERF=no_quad_lod) llvmpipe would choose differently. */
   if (gallivm_perf & GALLIVM_PERF_NO_QUAD_LOD) {
      fprintf(stderr, "gpgpusim sample: GALLIVM_PERF=no_quad_lod is not supported\n");
      abort();
   }
   smp.lp = lp;
   smp.last_valid = false; /* new draw: new resources */
   gpgpusim_setSampleCallback(sample_quad);
}

static void
drop_entries(bool (*match)(const struct sample_entry *, const void *), const void *data)
{
   if (!smp.initialized)
      return;
   struct util_dynarray keep;
   util_dynarray_init(&keep, NULL);
   util_dynarray_foreach(&smp.cache, struct sample_entry, e) {
      if (match(e, data))
         gallivm_destroy(e->gallivm);
      else
         util_dynarray_append(&keep, struct sample_entry, *e);
   }
   util_dynarray_fini(&smp.cache);
   smp.cache = keep;
   smp.last_valid = false;
}

static bool
entry_of_variant(const struct sample_entry *e, const void *data)
{
   const struct lp_fragment_shader_variant *v = data;
   return e->shader_no == v->shader->no && e->variant_no == v->no;
}

static bool
entry_of_context(const struct sample_entry *e, const void *data)
{
   return e->lp == data;
}

void
lp_gpgpusim_sample_forget_variant(const struct lp_fragment_shader_variant *variant)
{
   drop_entries(entry_of_variant, variant);
}

void
lp_gpgpusim_sample_forget_context(const struct llvmpipe_context *lp)
{
   drop_entries(entry_of_context, lp);
   if (smp.lp == lp)
      smp.lp = NULL;
}
