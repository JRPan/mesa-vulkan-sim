/*
 * llvmpipe's own blend/colormask/logic-op stage, callable from vulkan-sim
 * (fragment feed). Per FS variant + render-target format we JIT a function
 * around generate_unswizzled_blend (the code the FS JIT uses) that blends one
 * pixel of a 4x4 block; vulkan-sim calls it per fragment in rasterization order.
 * See docs/verification.md.
 */

#include "lp_gpgpusim_blend.h"
#include "lp_gpgpusim_feed.h"

#include "lp_context.h"
#include "lp_gpgpusim_verify.h"
#include "lp_jit.h"
#include "lp_rast.h"
#include "lp_setup_context.h"
#include "lp_state_fs.h"
#include "gallivm/lp_bld_init.h"
#include "gallivm/lp_bld_type.h"
#include "util/format/u_format.h"
#include "util/u_dual_blend.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../frontends/lavapipe/gpgpusim_calls_from_mesa.h"

typedef void (*lp_blend_fn)(struct lp_jit_context *context, uint8_t *color, int32_t stride,
                            const float *src /* [chan][16] */, const int32_t *mask /* [16] */);

struct blend_entry {
   const struct llvmpipe_context *lp;
   unsigned shader_no, variant_no; /* unique; variant pointers get reused */
   struct gallivm_state *gallivm;
   lp_blend_fn fn;
};

static struct {
   struct llvmpipe_context *lp;
   struct util_dynarray cache; /* struct blend_entry */
   bool initialized;
} blnd;

static lp_blend_fn
build_blend_fn(const struct lp_fragment_shader_variant *variant, struct gallivm_state **out_gallivm)
{
   struct gallivm_state *gallivm = gallivm_create("gpgpusim_blend", blnd.lp->context, NULL);
   LLVMContextRef lc = gallivm->context;
   LLVMBuilderRef b = gallivm->builder;

   struct lp_fragment_shader_variant types;
   memset(&types, 0, sizeof(types));
   types.gallivm = gallivm;
   lp_jit_init_types(&types);

   /* as generate_fragment: fs_type and the 16 x unorm8 blend vector type */
   struct lp_type fs_type;
   memset(&fs_type, 0, sizeof(fs_type));
   fs_type.floating = true;
   fs_type.sign = true;
   fs_type.width = 32;
   fs_type.length = MIN2(lp_native_vector_width / 32, 16);
   const unsigned num_fs = 16 / fs_type.length;
   struct lp_type blend_type;
   memset(&blend_type, 0, sizeof(blend_type));
   blend_type.norm = true;
   blend_type.width = 8;
   blend_type.length = 16;
   LLVMTypeRef blend_vec_type = lp_build_vec_type(gallivm, blend_type);

   LLVMTypeRef i8p = LLVMPointerType(LLVMInt8TypeInContext(lc), 0);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(lc);
   LLVMTypeRef args[5] = { types.jit_context_ptr_type, i8p, i32,
                           LLVMPointerType(LLVMFloatTypeInContext(lc), 0), LLVMPointerType(i32, 0) };
   LLVMValueRef func = LLVMAddFunction(gallivm->module, "gpgpusim_blend",
                                       LLVMFunctionType(LLVMVoidTypeInContext(lc), args, 5, 0));
   LLVMSetFunctionCallConv(func, LLVMCCallConv);
   LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(lc, func, "entry"));

   LLVMTypeRef fvec = lp_build_vec_type(gallivm, fs_type); /* src must be 32-byte aligned */
   struct lp_type mask_type = lp_int32_vec4_type();
   mask_type.length = fs_type.length;
   LLVMTypeRef mvec = lp_build_vec_type(gallivm, mask_type);

   LLVMValueRef fs_out_color[PIPE_MAX_COLOR_BUFS][TGSI_NUM_CHANNELS][4];
   memset(fs_out_color, 0, sizeof(fs_out_color));
   LLVMValueRef fs_mask[2 * 4];
   for (unsigned i = 0; i < num_fs; i++) {
      for (unsigned c = 0; c < TGSI_NUM_CHANNELS; c++) {
         LLVMValueRef idx = LLVMConstInt(i32, c * 16 + i * fs_type.length, 0);
         LLVMValueRef p = LLVMBuildGEP2(b, LLVMFloatTypeInContext(lc), LLVMGetParam(func, 3), &idx, 1, "");
         /* the blend code loads these itself (as from the FS's output allocas) */
         fs_out_color[0][c][i] = LLVMBuildBitCast(b, p, LLVMPointerType(fvec, 0), "");
      }
      LLVMValueRef idx = LLVMConstInt(i32, i * fs_type.length, 0);
      LLVMValueRef p = LLVMBuildGEP2(b, i32, LLVMGetParam(func, 4), &idx, 1, "");
      LLVMValueRef m = LLVMBuildLoad2(b, mvec, LLVMBuildBitCast(b, p, LLVMPointerType(mvec, 0), ""), "");
      LLVMSetAlignment(m, 4);
      fs_mask[i] = m;
   }

   LLVMValueRef color_ptr = LLVMBuildBitCast(b, LLVMGetParam(func, 1), LLVMPointerType(blend_vec_type, 0), "");
   generate_unswizzled_blend(gallivm, 0, (struct lp_fragment_shader_variant *)variant,
                             variant->key.cbuf_format[0], num_fs, fs_type, fs_mask, fs_out_color,
                             types.jit_context_type, LLVMGetParam(func, 0), blend_vec_type,
                             color_ptr, LLVMGetParam(func, 2), 1 /* per-pixel mask */, false);
   LLVMBuildRetVoid(b);

   gallivm_verify_function(gallivm, func);
   gallivm_compile_module(gallivm);
   lp_blend_fn fn = (lp_blend_fn)gallivm_jit_function(gallivm, func);
   gallivm_free_ir(gallivm);
   *out_gallivm = gallivm;
   return fn;
}

/* vulkan-sim callback: blends src into dst (float RGBA, the vulkan-sim FBO pixel). */
static void
blend_pixel(float dst[4], const float src[4])
{
   const struct lp_rast_state *state = lp_gpgpusim_feed_state();
   const struct lp_fragment_shader_variant *variant = state ? state->variant : NULL;
   const enum pipe_format fmt = variant ? variant->key.cbuf_format[0] : PIPE_FORMAT_NONE;
   if (!variant || util_format_get_blocksize(fmt) != 4 || util_format_is_pure_integer(fmt) ||
       variant->key.nr_cbufs != 1 || util_blend_state_is_dual(&variant->key.blend, 0)) {
      fprintf(stderr, "gpgpusim blend: unsupported target (format %s, %u cbufs); only a single "
              "4-byte normalized colour buffer is handled\n", util_format_name(fmt),
              variant ? variant->key.nr_cbufs : 0);
      abort();
   }

   lp_blend_fn fn = NULL;
   util_dynarray_foreach(&blnd.cache, struct blend_entry, e) {
      if (e->shader_no == variant->shader->no && e->variant_no == variant->no) {
         fn = e->fn;
         break;
      }
   }
   if (!fn) {
      struct blend_entry e = { blnd.lp, variant->shader->no, variant->no, NULL, NULL };
      e.fn = fn = build_blend_fn(variant, &e.gallivm);
      util_dynarray_append(&blnd.cache, struct blend_entry, e);
   }
   /* 4x4 block, 4 bytes per pixel; the pixel sits at lane 0 = block (0,0) */
   uint8_t block[4 * 16] __attribute__((aligned(16)));
   memset(block, 0, sizeof(block));
   /* dst is always an unpack of stored bytes (seeded or blended), so the exact
    * util_format pack recovers them; llvmpipe's approximate sRGB encode might not */
   util_format_pack_rgba(fmt, block, dst, 1);
   float s[4 * 16] __attribute__((aligned(64)));
   int32_t mask[16] __attribute__((aligned(16)));
   memset(s, 0, sizeof(s));
   memset(mask, 0, sizeof(mask));
   for (unsigned c = 0; c < 4; c++)
      s[c * 16] = src[c];
   mask[0] = ~0;
   fn((struct lp_jit_context *)&state->jit_context, block, 16, s, mask);
   util_format_unpack_rgba(fmt, dst, block, 1);
}

void
lp_gpgpusim_blend_begin(struct llvmpipe_context *lp)
{
   if (!blnd.initialized) {
      blnd.initialized = true;
      util_dynarray_init(&blnd.cache, NULL);
   }
   blnd.lp = lp;
   /* Only route stores through llvmpipe's blend stage when it changes anything. */
   const struct pipe_blend_state *bs = lp->blend;
   const bool needed = bs && (bs->logicop_enable || bs->rt[0].blend_enable || bs->rt[0].colormask != 0xf);
   if (needed)
      printf("gpgpusim blend: draw uses %s%s%s; FBO stores go through llvmpipe's blend stage\n",
             bs->rt[0].blend_enable ? "blending " : "", bs->logicop_enable ? "logic op " : "",
             bs->rt[0].colormask != 0xf ? "colour mask" : "");
   gpgpusim_setBlendCallback(needed ? blend_pixel : NULL);
}

static void
drop_blend_entries(const struct llvmpipe_context *lp, const struct lp_fragment_shader_variant *variant)
{
   if (!blnd.initialized)
      return;
   struct util_dynarray keep;
   util_dynarray_init(&keep, NULL);
   util_dynarray_foreach(&blnd.cache, struct blend_entry, e) {
      const bool match = variant ? (e->shader_no == variant->shader->no && e->variant_no == variant->no)
                                 : e->lp == lp;
      if (match)
         gallivm_destroy(e->gallivm);
      else
         util_dynarray_append(&keep, struct blend_entry, *e);
   }
   util_dynarray_fini(&blnd.cache);
   blnd.cache = keep;
}

void
lp_gpgpusim_blend_forget_variant(const struct lp_fragment_shader_variant *variant)
{
   drop_blend_entries(NULL, variant);
}

void
lp_gpgpusim_blend_forget_context(const struct llvmpipe_context *lp)
{
   drop_blend_entries(lp, NULL);
   if (blnd.lp == lp)
      blnd.lp = NULL;
}
