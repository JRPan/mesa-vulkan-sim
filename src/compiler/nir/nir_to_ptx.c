/*
 * NIR -> PTX for vulkan-sim graphics shaders (VS/FS), in a single pass.
 *
 * Emits the final PTX that vulkan-sim parses: scalar registers, explicit
 * address arithmetic for derefs, and vulkan-sim pseudo-ops (rt_alloc_mem,
 * load_ubo, tex, ...). Anything not handled aborts with the offending NIR
 * instruction, rather than emitting PTX that fails or misbehaves later.
 */

#include "nir_to_ptx.h"
#include "nir.h"
#include "compiler/nir_types.h"
#include "compiler/shader_enums.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/u_math.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PTX_MAX_LOOP_DEPTH 64

typedef struct {
   FILE *fp;
   nir_shader *shader;
   void *mem_ctx;
   struct hash_table *var_names; /* nir_variable * -> sanitized name */
   struct set *used_names;
   struct set *derefed_vars; /* variables some deref_var references */
   unsigned next_label;
   unsigned next_tmp;
   unsigned loop_stack[PTX_MAX_LOOP_DEPTH];
   unsigned loop_depth;
} ptx_ctx;

static void NORETURN PRINTFLIKE(2, 3)
ptx_fail(const nir_instr *instr, const char *fmt, ...)
{
   va_list args;
   fprintf(stderr, "nir_to_ptx: ");
   va_start(args, fmt);
   vfprintf(stderr, fmt, args);
   va_end(args);
   fprintf(stderr, "\n");
   if (instr) {
      fprintf(stderr, "nir_to_ptx: at: ");
      nir_print_instr(instr, stderr);
      fprintf(stderr, "\n");
   }
   abort();
}

#define ptx_check(cond, instr, ...) \
   do { if (!(cond)) ptx_fail(instr, __VA_ARGS__); } while (0)

/* Register names, formatted into a small ring so several fit in one fprintf. */
static const char *
reg_name(const nir_ssa_def *def, unsigned comp)
{
   static char ring[16][32];
   static unsigned slot = 0;
   char *buf = ring[slot++ % ARRAY_SIZE(ring)];
   if (def->num_components == 1)
      snprintf(buf, sizeof(ring[0]), "%%ssa_%u", def->index);
   else
      snprintf(buf, sizeof(ring[0]), "%%ssa_%u_%u", def->index, comp);
   return buf;
}

static const char *
phi_name(const nir_ssa_def *def, unsigned comp)
{
   static char ring[8][32];
   static unsigned slot = 0;
   char *buf = ring[slot++ % ARRAY_SIZE(ring)];
   snprintf(buf, sizeof(ring[0]), "%%phi_%u_%u", def->index, comp);
   return buf;
}

static const char *
src_name(const nir_src *src, unsigned comp, const nir_instr *instr)
{
   ptx_check(src->is_ssa, instr, "non-SSA source (expected SSA-form NIR)");
   ptx_check(comp < src->ssa->num_components, instr,
             "source component %u out of range (%u components)",
             comp, src->ssa->num_components);
   return reg_name(src->ssa, comp);
}

static const char *
alu_src(const nir_alu_instr *alu, unsigned i, unsigned comp)
{
   return src_name(&alu->src[i].src, alu->src[i].swizzle[comp], &alu->instr);
}

/* Declares a scratch register inline and returns its name. */
static const char *
new_tmp(ptx_ctx *ctx, const char *type)
{
   static char ring[8][32];
   static unsigned slot = 0;
   char *buf = ring[slot++ % ARRAY_SIZE(ring)];
   snprintf(buf, sizeof(ring[0]), "%%tmp_%u", ctx->next_tmp++);
   fprintf(ctx->fp, "\t.reg .%s %s;\n", type, buf);
   return buf;
}

static bool
def_is_pointer(const nir_ssa_def *def)
{
   const nir_instr *instr = def->parent_instr;
   if (instr->type == nir_instr_type_deref)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
      return op == nir_intrinsic_vulkan_resource_index ||
             op == nir_intrinsic_load_vulkan_descriptor;
   }
   return false;
}

static void
decl_def_regs(ptx_ctx *ctx, const nir_ssa_def *def, const char *(*name)(const nir_ssa_def *, unsigned))
{
   char type[8];
   if (def_is_pointer(def))
      snprintf(type, sizeof(type), "b64");
   else if (def->bit_size == 1)
      snprintf(type, sizeof(type), "pred");
   else
      snprintf(type, sizeof(type), "b%u", def->bit_size);
   for (unsigned c = 0; c < def->num_components; c++)
      fprintf(ctx->fp, "\t.reg .%s %s;\n", type, name(def, c));
}

static bool
decl_def_cb(nir_ssa_def *def, void *data)
{
   decl_def_regs(data, def, reg_name);
   return true;
}

/* ---- variables ---------------------------------------------------------- */

static const char *
var_name(ptx_ctx *ctx, nir_variable *var)
{
   struct hash_entry *entry = _mesa_hash_table_search(ctx->var_names, var);
   if (entry)
      return entry->data;

   const char *orig = var->name ? var->name : "";
   char *name = ralloc_strdup(ctx->mem_ctx, orig);
   bool valid = name[0] != '\0' && !(name[0] >= '0' && name[0] <= '9');
   for (char *p = name; *p; p++) {
      if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
            (*p >= '0' && *p <= '9') || *p == '_')) {
         *p = '_';
         valid = false;
      }
   }
   if (!valid || _mesa_set_search(ctx->used_names, name))
      name = ralloc_asprintf(ctx->mem_ctx, "var%u_%s",
                             ctx->used_names->entries, name);
   _mesa_set_add(ctx->used_names, name);
   _mesa_hash_table_insert(ctx->var_names, var, name);
   return name;
}

const char *
nir_to_ptx_var_location(const nir_shader *shader, const nir_variable *var, char *buf, size_t size)
{
   const char *loc = NULL;
   gl_shader_stage stage = shader->info.stage;
   if (var->data.mode == nir_var_shader_in) {
      loc = stage == MESA_SHADER_VERTEX ? gl_vert_attrib_name(var->data.location)
                                        : gl_varying_slot_name_for_stage(var->data.location, stage);
   } else if (var->data.mode == nir_var_shader_out) {
      loc = stage == MESA_SHADER_FRAGMENT ? gl_frag_result_name(var->data.location)
                                          : gl_varying_slot_name_for_stage(var->data.location, stage);
   }
   if (!loc) {
      if (var->data.location == ~0)
         return "~0";
      snprintf(buf, size, "%u", var->data.location);
      return buf;
   }

   /* vulkan-sim matches VS outputs to FS inputs on "<slot>_<components>". */
   unsigned n = glsl_get_components(glsl_without_array(var->type));
   if ((var->data.mode == nir_var_shader_in || var->data.mode == nir_var_shader_out) &&
       n != 0 && n < 16) {
      static const char xyzw[] = "xyzw";
      int len = snprintf(buf, size, "%s_", loc);
      for (unsigned i = 0; i < n; i++) {
         unsigned comp = i + var->data.location_frac;
         buf[len++] = comp < 4 ? xyzw[comp] : 'a' + (comp - 4);
      }
      buf[len] = '\0';
      return buf;
   }
   return loc;
}

static void
emit_var_decl(ptx_ctx *ctx, nir_variable *var)
{
   unsigned size = 0, align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);
   if (size < 4)
      size = 4;
   /* vulkan-sim (CRISP) expects size * vector_elements for vector I/O. */
   unsigned vec = glsl_get_vector_elements(var->type);
   unsigned alloc = vec ? size * vec : size;

   unsigned set = var->data.remapped ? var->data.original_descriptor_set : var->data.descriptor_set;
   unsigned binding = var->data.remapped ? var->data.original_binding : var->data.binding;
   const char *name = var_name(ctx, var);
   char locbuf[64];

   /* Per-invocation locals live in PTX local memory; derefs take their address. */
   if (var->data.mode == nir_var_function_temp || var->data.mode == nir_var_shader_temp) {
      fprintf(ctx->fp, "\t.local .align %u .b8 %s[%u];\n", MAX2(align, 4), name, size);
      return;
   }

   /* Resources reached only through indices/offsets (tex, load_ubo,
    * load_push_constant, ...) need no declaration, and vulkan-sim cannot
    * resolve every descriptor type or a "~0" location for every mode. */
   const nir_variable_mode resource_modes = nir_var_uniform | nir_var_image | nir_var_mem_ubo |
                                            nir_var_mem_ssbo | nir_var_mem_push_const;
   if ((var->data.mode & resource_modes) && !_mesa_set_search(ctx->derefed_vars, var))
      return;

   fprintf(ctx->fp, "\t.reg .b64 %%%s;\n", name);
   if (var->data.mode == nir_var_uniform || var->data.mode == nir_var_image) {
      fprintf(ctx->fp, "\tload_vulkan_descriptor %%%s, %u, %u;\n", name, set, binding);
   } else {
      fprintf(ctx->fp, "\trt_alloc_mem %%%s, %u, %u, %u, %s;\n", name, alloc,
              (unsigned)var->data.mode, var->data.driver_location,
              nir_to_ptx_var_location(ctx->shader, var, locbuf, sizeof(locbuf)));
   }
}

/* ---- load_const --------------------------------------------------------- */

static void
emit_load_const(ptx_ctx *ctx, nir_load_const_instr *lc)
{
   for (unsigned c = 0; c < lc->def.num_components; c++) {
      const char *dst = reg_name(&lc->def, c);
      switch (lc->def.bit_size) {
      case 1: {
         const char *t = new_tmp(ctx, "u32");
         fprintf(ctx->fp, "\tmov.u32 %s, %u;\n", t, lc->value[c].b ? 1 : 0);
         fprintf(ctx->fp, "\tsetp.ne.u32 %s, %s, 0;\n", dst, t);
         break;
      }
      case 16:
         fprintf(ctx->fp, "\tmov.b16 %s, 0x%04x;\n", dst, lc->value[c].u16);
         break;
      case 32:
         fprintf(ctx->fp, "\tmov.b32 %s, 0x%08x;\n", dst, lc->value[c].u32);
         break;
      case 64:
         fprintf(ctx->fp, "\tmov.b64 %s, 0d%016" PRIx64 ";\n", dst, lc->value[c].u64);
         break;
      default:
         ptx_fail(&lc->instr, "unsupported load_const bit size %u", lc->def.bit_size);
      }
   }
}

/* ---- ALU ---------------------------------------------------------------- */

static void
require_bits(const nir_alu_instr *alu, unsigned dst_bits, unsigned src_bits)
{
   ptx_check(alu->dest.dest.ssa.bit_size == dst_bits, &alu->instr,
             "expected %u-bit destination", dst_bits);
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
      ptx_check(nir_src_bit_size(alu->src[i].src) == src_bits, &alu->instr,
                "expected %u-bit source %u", src_bits, i);
}

/* PTX type suffix for an integer op of the given width and signedness. */
static const char *
int_type(const nir_alu_instr *alu, unsigned bits, bool is_signed)
{
   switch (bits) {
   case 16: return is_signed ? "s16" : "u16";
   case 32: return is_signed ? "s32" : "u32";
   case 64: return is_signed ? "s64" : "u64";
   default: ptx_fail(&alu->instr, "unsupported integer bit size %u", bits);
   }
}

static void
emit_alu(ptx_ctx *ctx, nir_alu_instr *alu)
{
   FILE *fp = ctx->fp;
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_ssa_def *d = &alu->dest.dest.ssa;
   const nir_instr *in = &alu->instr;

   ptx_check(alu->dest.dest.is_ssa, in, "non-SSA destination");
   ptx_check(!alu->dest.saturate, in, "saturate modifier not supported");
   for (unsigned i = 0; i < info->num_inputs; i++)
      ptx_check(!alu->src[i].negate && !alu->src[i].abs, in, "source modifiers not supported");

   unsigned bits = d->bit_size;
   unsigned sbits = info->num_inputs ? nir_src_bit_size(alu->src[0].src) : 0;

   if (nir_op_is_vec(alu->op)) {
      for (unsigned c = 0; c < d->num_components; c++) {
         if (bits == 1)
            fprintf(fp, "\tmov.pred %s, %s;\n", reg_name(d, c), alu_src(alu, c, 0));
         else
            fprintf(fp, "\tmov.b%u %s, %s;\n", bits, reg_name(d, c), alu_src(alu, c, 0));
      }
      return;
   }

   /* Horizontal ops: vector inputs, scalar output. */
   if (alu->op == nir_op_fsum3 || alu->op == nir_op_fsum4) {
      require_bits(alu, 32, 32);
      unsigned n = alu->op == nir_op_fsum3 ? 3 : 4;
      const char *dst = reg_name(d, 0);
      fprintf(fp, "\tadd.f32 %s, %s, %s;\n", dst, alu_src(alu, 0, 0), alu_src(alu, 0, 1));
      for (unsigned i = 2; i < n; i++)
         fprintf(fp, "\tadd.f32 %s, %s, %s;\n", dst, dst, alu_src(alu, 0, i));
      return;
   }

   ptx_check(info->output_size == 0, in, "unsupported non-per-component ALU op %s", info->name);

   /* 32-bit-boolean compares (true = ~0): compare into a predicate, then select. */
   nir_op op = alu->op;
   switch (op) {
   case nir_op_flt32:  op = nir_op_flt;  break;
   case nir_op_fge32:  op = nir_op_fge;  break;
   case nir_op_feq32:  op = nir_op_feq;  break;
   case nir_op_fneu32: op = nir_op_fneu; break;
   case nir_op_ilt32:  op = nir_op_ilt;  break;
   case nir_op_ige32:  op = nir_op_ige;  break;
   case nir_op_ult32:  op = nir_op_ult;  break;
   case nir_op_uge32:  op = nir_op_uge;  break;
   case nir_op_ieq32:  op = nir_op_ieq;  break;
   case nir_op_ine32:  op = nir_op_ine;  break;
   default: break;
   }
   const bool bool32 = op != alu->op;
   const unsigned cmp_bits = bool32 ? 32 : 1;

   for (unsigned c = 0; c < d->num_components; c++) {
      const char *dst = reg_name(d, c);
      const char *cmp_dst = bool32 ? new_tmp(ctx, "pred") : dst;
      const char *a = info->num_inputs > 0 ? alu_src(alu, 0, c) : NULL;
      const char *b = info->num_inputs > 1 ? alu_src(alu, 1, c) : NULL;
      const char *s2 = info->num_inputs > 2 ? alu_src(alu, 2, c) : NULL;

#define F32_UNOP(op_str)  require_bits(alu, 32, 32); fprintf(fp, "\t" op_str " %s, %s;\n", dst, a); break
#define F32_BINOP(op_str) require_bits(alu, 32, 32); fprintf(fp, "\t" op_str " %s, %s, %s;\n", dst, a, b); break
#define F32_CMP(cmp)      require_bits(alu, cmp_bits, 32); fprintf(fp, "\tsetp." cmp ".f32 %s, %s, %s;\n", cmp_dst, a, b); break

      switch (op) {
      case nir_op_mov:
         if (bits == 1)
            fprintf(fp, "\tmov.pred %s, %s;\n", dst, a);
         else
            fprintf(fp, "\tmov.b%u %s, %s;\n", bits, dst, a);
         break;

      /* float arithmetic */
      case nir_op_fneg:   F32_UNOP("neg.f32");
      case nir_op_fabs:   F32_UNOP("abs.f32");
      case nir_op_fadd:   F32_BINOP("add.f32");
      case nir_op_fsub:   F32_BINOP("sub.f32");
      case nir_op_fmul:   F32_BINOP("mul.f32");
      case nir_op_fdiv:   F32_BINOP("div.rn.f32");
      case nir_op_fmin:   F32_BINOP("min.f32");
      case nir_op_fmax:   F32_BINOP("max.f32");
      case nir_op_frcp:   F32_UNOP("rcp.approx.f32");
      case nir_op_frsq:   F32_UNOP("rsqrt.approx.f32");
      case nir_op_fsqrt:  F32_UNOP("sqrt.approx.f32");
      case nir_op_fexp2:  F32_UNOP("ex2.approx.f32");
      case nir_op_flog2:  F32_UNOP("lg2.approx.f32");
      case nir_op_fsin:   F32_UNOP("sin.approx.f32");
      case nir_op_fcos:   F32_UNOP("cos.approx.f32");
      case nir_op_ffloor: F32_UNOP("cvt.rmi.f32.f32");
      case nir_op_fceil:  F32_UNOP("cvt.rpi.f32.f32");
      case nir_op_ftrunc: F32_UNOP("cvt.rzi.f32.f32");
      case nir_op_fround_even: F32_UNOP("cvt.rni.f32.f32");
      case nir_op_ffma:
         require_bits(alu, 32, 32);
         fprintf(fp, "\tfma.rn.f32 %s, %s, %s, %s;\n", dst, a, b, s2);
         break;
      case nir_op_ffract: {
         require_bits(alu, 32, 32);
         const char *t = new_tmp(ctx, "f32");
         fprintf(fp, "\tcvt.rmi.f32.f32 %s, %s;\n", t, a);
         fprintf(fp, "\tsub.f32 %s, %s, %s;\n", dst, a, t);
         break;
      }
      case nir_op_fpow:
         require_bits(alu, 32, 32);
         fprintf(fp, "\tlg2.approx.f32 %s, %s;\n", dst, a);
         fprintf(fp, "\tmul.f32 %s, %s, %s;\n", dst, dst, b);
         fprintf(fp, "\tex2.approx.f32 %s, %s;\n", dst, dst);
         break;
      case nir_op_fsat:
         require_bits(alu, 32, 32);
         fprintf(fp, "\tmax.f32 %s, %s, 0f00000000;\n", dst, a);
         fprintf(fp, "\tmin.f32 %s, %s, 0f3F800000;\n", dst, dst);
         break;
      case nir_op_flrp: { /* a * (1 - c) + b * c */
         require_bits(alu, 32, 32);
         const char *t = new_tmp(ctx, "f32");
         fprintf(fp, "\tsub.f32 %s, 0f3F800000, %s;\n", t, s2);
         fprintf(fp, "\tmul.f32 %s, %s, %s;\n", t, a, t);
         fprintf(fp, "\tmul.f32 %s, %s, %s;\n", dst, b, s2);
         fprintf(fp, "\tadd.f32 %s, %s, %s;\n", dst, dst, t);
         break;
      }
      case nir_op_fsign: { /* NaN -> 0, +-0 -> itself, else +-1 */
         require_bits(alu, 32, 32);
         const char *p = new_tmp(ctx, "pred");
         fprintf(fp, "\tmov.b32 %s, %s;\n", dst, a);
         fprintf(fp, "\tsetp.gt.f32 %s, %s, 0f00000000;\n", p, a);
         fprintf(fp, "\tselp.f32 %s, 0f3F800000, %s, %s;\n", dst, dst, p);
         fprintf(fp, "\tsetp.lt.f32 %s, %s, 0f00000000;\n", p, a);
         fprintf(fp, "\tselp.f32 %s, 0fBF800000, %s, %s;\n", dst, dst, p);
         fprintf(fp, "\tsetp.nan.f32 %s, %s, %s;\n", p, a, a);
         fprintf(fp, "\tselp.f32 %s, 0f00000000, %s, %s;\n", dst, dst, p);
         break;
      }

      /* float comparisons; fneu is unordered (true on NaN) */
      case nir_op_flt:  F32_CMP("lt");
      case nir_op_fge:  F32_CMP("ge");
      case nir_op_feq:  F32_CMP("eq");
      case nir_op_fneu: F32_CMP("neu");

      /* integer comparisons */
      case nir_op_ilt: case nir_op_ige: case nir_op_ult: case nir_op_uge: {
         ptx_check(bits == cmp_bits, in, "expected %u-bit boolean result", cmp_bits);
         bool is_signed = op == nir_op_ilt || op == nir_op_ige;
         const char *cmp = (op == nir_op_ilt || op == nir_op_ult) ? "lt" : "ge";
         fprintf(fp, "\tsetp.%s.%s %s, %s, %s;\n", cmp, int_type(alu, sbits, is_signed), cmp_dst, a, b);
         break;
      }
      case nir_op_ieq: case nir_op_ine:
         ptx_check(bits == cmp_bits, in, "expected %u-bit boolean result", cmp_bits);
         if (sbits == 1) {
            fprintf(fp, "\txor.pred %s, %s, %s;\n", cmp_dst, a, b);
            if (op == nir_op_ieq)
               fprintf(fp, "\tnot.pred %s, %s;\n", cmp_dst, cmp_dst);
         } else {
            fprintf(fp, "\tsetp.%s.%s %s, %s, %s;\n", op == nir_op_ieq ? "eq" : "ne",
                    int_type(alu, sbits, false), cmp_dst, a, b);
         }
         break;

      /* integer arithmetic */
      case nir_op_iadd: fprintf(fp, "\tadd.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;
      case nir_op_isub: fprintf(fp, "\tsub.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;
      case nir_op_amul: /* address multiply = imul */
      case nir_op_imul: fprintf(fp, "\tmul.lo.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;
      case nir_op_ineg: fprintf(fp, "\tneg.%s %s, %s;\n", int_type(alu, bits, true), dst, a); break;
      case nir_op_iabs: fprintf(fp, "\tabs.%s %s, %s;\n", int_type(alu, bits, true), dst, a); break;
      case nir_op_imin: fprintf(fp, "\tmin.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;
      case nir_op_imax: fprintf(fp, "\tmax.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;
      case nir_op_umin: fprintf(fp, "\tmin.%s %s, %s, %s;\n", int_type(alu, bits, false), dst, a, b); break;
      case nir_op_umax: fprintf(fp, "\tmax.%s %s, %s, %s;\n", int_type(alu, bits, false), dst, a, b); break;
      case nir_op_idiv: fprintf(fp, "\tdiv.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;
      case nir_op_udiv: fprintf(fp, "\tdiv.%s %s, %s, %s;\n", int_type(alu, bits, false), dst, a, b); break;
      case nir_op_irem: fprintf(fp, "\trem.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;
      case nir_op_umod: fprintf(fp, "\trem.%s %s, %s, %s;\n", int_type(alu, bits, false), dst, a, b); break;
      case nir_op_umul_high: fprintf(fp, "\tmul.hi.%s %s, %s, %s;\n", int_type(alu, bits, false), dst, a, b); break;
      case nir_op_imul_high: fprintf(fp, "\tmul.hi.%s %s, %s, %s;\n", int_type(alu, bits, true), dst, a, b); break;

      /* NIR masks the shift count to the bit size; PTX clamps it instead. */
      case nir_op_ishl: case nir_op_ishr: case nir_op_ushr: {
         const char *t = new_tmp(ctx, "b32");
         fprintf(fp, "\tand.b32 %s, %s, %u;\n", t, b, bits - 1);
         if (alu->op == nir_op_ishl)
            fprintf(fp, "\tshl.b%u %s, %s, %s;\n", bits, dst, a, t);
         else
            fprintf(fp, "\tshr.%s %s, %s, %s;\n", int_type(alu, bits, alu->op == nir_op_ishr), dst, a, t);
         break;
      }

      /* bitwise / boolean logic */
      case nir_op_iand: case nir_op_ior: case nir_op_ixor: {
         const char *op = alu->op == nir_op_iand ? "and" : alu->op == nir_op_ior ? "or" : "xor";
         if (bits == 1)
            fprintf(fp, "\t%s.pred %s, %s, %s;\n", op, dst, a, b);
         else
            fprintf(fp, "\t%s.b%u %s, %s, %s;\n", op, bits, dst, a, b);
         break;
      }
      case nir_op_inot:
         if (bits == 1)
            fprintf(fp, "\tnot.pred %s, %s;\n", dst, a);
         else
            fprintf(fp, "\tnot.b%u %s, %s;\n", bits, dst, a);
         break;
      case nir_op_b32csel: {
         ptx_check(sbits == 32 && bits != 1, in, "b32csel: expected a 32-bit condition");
         const char *p = new_tmp(ctx, "pred");
         fprintf(fp, "\tsetp.ne.b32 %s, %s, 0;\n", p, a);
         fprintf(fp, "\tselp.b%u %s, %s, %s, %s;\n", bits, dst, b, s2, p);
         break;
      }
      case nir_op_bcsel:
         ptx_check(sbits == 1, in, "bcsel condition must be a 1-bit boolean");
         if (bits == 1) {
            const char *t0 = new_tmp(ctx, "pred");
            const char *t1 = new_tmp(ctx, "pred");
            fprintf(fp, "\tand.pred %s, %s, %s;\n", t0, a, b);
            fprintf(fp, "\tnot.pred %s, %s;\n", t1, a);
            fprintf(fp, "\tand.pred %s, %s, %s;\n", t1, t1, s2);
            fprintf(fp, "\tor.pred %s, %s, %s;\n", dst, t0, t1);
         } else {
            fprintf(fp, "\tselp.b%u %s, %s, %s, %s;\n", bits, dst, b, s2, a);
         }
         break;

      /* conversions; float->int truncates (C cast) */
      case nir_op_f2i32: require_bits(alu, 32, 32); fprintf(fp, "\tcvt.rzi.s32.f32 %s, %s;\n", dst, a); break;
      case nir_op_f2u32: require_bits(alu, 32, 32); fprintf(fp, "\tcvt.rzi.u32.f32 %s, %s;\n", dst, a); break;
      case nir_op_i2f32:
         ptx_check(bits == 32, in, "expected 32-bit destination");
         fprintf(fp, "\tcvt.rn.f32.%s %s, %s;\n", int_type(alu, sbits, true), dst, a);
         break;
      case nir_op_u2f32:
         ptx_check(bits == 32, in, "expected 32-bit destination");
         fprintf(fp, "\tcvt.rn.f32.%s %s, %s;\n", int_type(alu, sbits, false), dst, a);
         break;
      case nir_op_f2f32: require_bits(alu, 32, 32); fprintf(fp, "\tmov.b32 %s, %s;\n", dst, a); break;
      case nir_op_b2f32:
      case nir_op_b2i32: {
         ptx_check((sbits == 1 || sbits == 32) && bits == 32, in, "expected 1- or 32-bit bool -> 32-bit");
         const char *p = a;
         if (sbits == 32) {
            p = new_tmp(ctx, "pred");
            fprintf(fp, "\tsetp.ne.b32 %s, %s, 0;\n", p, a);
         }
         if (op == nir_op_b2f32)
            fprintf(fp, "\tselp.f32 %s, 0f3F800000, 0f00000000, %s;\n", dst, p);
         else
            fprintf(fp, "\tselp.u32 %s, 1, 0, %s;\n", dst, p);
         break;
      }
      case nir_op_i2i32: case nir_op_u2u32: case nir_op_i2i64: case nir_op_u2u64: {
         bool is_signed = alu->op == nir_op_i2i32 || alu->op == nir_op_i2i64;
         if (sbits == bits)
            fprintf(fp, "\tmov.b%u %s, %s;\n", bits, dst, a);
         else
            fprintf(fp, "\tcvt.%s.%s %s, %s;\n", int_type(alu, bits, is_signed),
                    int_type(alu, sbits, is_signed), dst, a);
         break;
      }
      case nir_op_pack_64_2x32_split: {
         require_bits(alu, 64, 32);
         const char *t = new_tmp(ctx, "b64");
         fprintf(fp, "\tcvt.u64.u32 %s, %s;\n", t, b);
         fprintf(fp, "\tshl.b64 %s, %s, 32;\n", t, t);
         fprintf(fp, "\tcvt.u64.u32 %s, %s;\n", dst, a);
         fprintf(fp, "\tor.b64 %s, %s, %s;\n", dst, dst, t);
         break;
      }
      case nir_op_unpack_64_2x32_split_x:
         require_bits(alu, 32, 64);
         fprintf(fp, "\tcvt.u32.u64 %s, %s;\n", dst, a);
         break;
      case nir_op_unpack_64_2x32_split_y: {
         require_bits(alu, 32, 64);
         const char *t = new_tmp(ctx, "b64");
         fprintf(fp, "\tshr.u64 %s, %s, 32;\n", t, a);
         fprintf(fp, "\tcvt.u32.u64 %s, %s;\n", dst, t);
         break;
      }

      default:
         ptx_fail(in, "unsupported ALU op %s", info->name);
      }
      if (bool32)
         fprintf(fp, "\tselp.b32 %s, 0xffffffff, 0, %s;\n", dst, cmp_dst);
#undef F32_UNOP
#undef F32_BINOP
#undef F32_CMP
   }
}

/* ---- derefs ------------------------------------------------------------- */

/* Explicit layout offset if the type has one, else natural alignment (as
 * glsl_get_natural_size_align_bytes lays out the struct for rt_alloc_mem). */
static int
struct_field_offset(const struct glsl_type *type, unsigned index)
{
   const int explicit_offset = glsl_get_struct_field_offset(type, index);
   if (explicit_offset >= 0)
      return explicit_offset;
   unsigned offset = 0;
   for (unsigned i = 0; i <= index; i++) {
      unsigned size = 0, align = 0;
      glsl_get_natural_size_align_bytes(glsl_get_struct_field(type, i), &size, &align);
      offset = ALIGN_POT(offset, MAX2(align, 1));
      if (i == index)
         break;
      offset += size;
   }
   return offset;
}

static bool
deref_is_local(const nir_deref_instr *deref)
{
   return nir_deref_mode_is_one_of(deref, nir_var_function_temp | nir_var_shader_temp);
}

static unsigned
deref_array_stride(const nir_deref_instr *deref)
{
   /* nir_deref_instr_array_stride asserts on matrices without an explicit
    * layout (it queries row-major-ness); those use the natural column size. */
   const nir_deref_instr *parent = nir_deref_instr_parent(deref);
   const bool implicit_matrix = deref->deref_type == nir_deref_type_array && parent &&
                                glsl_type_is_matrix(parent->type) &&
                                glsl_get_explicit_stride(parent->type) == 0;
   unsigned stride = implicit_matrix ? 0 : nir_deref_instr_array_stride((nir_deref_instr *)deref);
   if (stride == 0) {
      /* No explicit layout (locals, I/O): natural size, as rt_alloc_mem sizes them. */
      unsigned size = 0, align = 0;
      glsl_get_natural_size_align_bytes(deref->type, &size, &align);
      stride = ALIGN_POT(size, MAX2(align, 1));
   }
   ptx_check(stride != 0, &deref->instr, "zero array stride");
   return stride;
}

static void
emit_deref(ptx_ctx *ctx, nir_deref_instr *deref)
{
   FILE *fp = ctx->fp;
   const char *dst = reg_name(&deref->dest.ssa, 0);
   ptx_check(deref->dest.is_ssa, &deref->instr, "non-SSA deref");

   switch (deref->deref_type) {
   case nir_deref_type_var:
      if (deref_is_local(deref))
         fprintf(fp, "\tmov.u64 %s, %s;\n", dst, var_name(ctx, deref->var));
      else
         fprintf(fp, "\tmov.b64 %s, %%%s;\n", dst, var_name(ctx, deref->var));
      break;
   case nir_deref_type_cast:
      fprintf(fp, "\tmov.b64 %s, %s;\n", dst, src_name(&deref->parent, 0, &deref->instr));
      break;
   case nir_deref_type_struct: {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      int offset = struct_field_offset(parent->type, deref->strct.index);
      fprintf(fp, "\tadd.u64 %s, %s, %d;\n", dst, src_name(&deref->parent, 0, &deref->instr), offset);
      break;
   }
   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array: {
      unsigned stride = deref_array_stride(deref);
      const char *parent = src_name(&deref->parent, 0, &deref->instr);
      if (nir_src_is_const(deref->arr.index)) {
         int64_t off = nir_src_as_int(deref->arr.index) * (int64_t)stride;
         fprintf(fp, "\tadd.s64 %s, %s, %" PRId64 ";\n", dst, parent, off);
      } else {
         ptx_check(nir_src_bit_size(deref->arr.index) == 32, &deref->instr, "expected 32-bit array index");
         const char *t = new_tmp(ctx, "b64");
         fprintf(fp, "\tmul.wide.s32 %s, %s, %u;\n", t, src_name(&deref->arr.index, 0, &deref->instr), stride);
         fprintf(fp, "\tadd.s64 %s, %s, %s;\n", dst, parent, t);
      }
      break;
   }
   default:
      ptx_fail(&deref->instr, "unsupported deref type %d", deref->deref_type);
   }
}

/* ---- intrinsics --------------------------------------------------------- */

static void
emit_intrinsic(ptx_ctx *ctx, nir_intrinsic_instr *intr)
{
   FILE *fp = ctx->fp;
   const nir_instr *in = &intr->instr;
   nir_ssa_def *d = nir_intrinsic_infos[intr->intrinsic].has_dest ? &intr->dest.ssa : NULL;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_deref: {
      const char *ptr = src_name(&intr->src[0], 0, in);
      const char *space = deref_is_local(nir_src_as_deref(intr->src[0])) ? "local" : "global";
      for (unsigned c = 0; c < d->num_components; c++) {
         if (d->bit_size == 1) {
            const char *t = new_tmp(ctx, "u32");
            fprintf(fp, "\tld.%s.u32 %s, [%s + %u];\n", space, t, ptr, c * 4);
            fprintf(fp, "\tsetp.ne.u32 %s, %s, 0;\n", reg_name(d, c), t);
         } else {
            fprintf(fp, "\tld.%s.b%u %s, [%s + %u];\n", space, d->bit_size, reg_name(d, c), ptr,
                    c * d->bit_size / 8);
         }
      }
      break;
   }
   case nir_intrinsic_store_deref: {
      const char *ptr = src_name(&intr->src[0], 0, in);
      const char *space = deref_is_local(nir_src_as_deref(intr->src[0])) ? "local" : "global";
      unsigned mask = nir_intrinsic_write_mask(intr);
      unsigned vbits = nir_src_bit_size(intr->src[1]);
      for (unsigned c = 0; c < nir_src_num_components(intr->src[1]); c++) {
         if (!(mask & (1u << c)))
            continue;
         const char *v = src_name(&intr->src[1], c, in);
         if (vbits == 1) {
            const char *t = new_tmp(ctx, "u32");
            fprintf(fp, "\tselp.u32 %s, 1, 0, %s;\n", t, v);
            fprintf(fp, "\tst.%s.u32 [%s + %u], %s;\n", space, ptr, c * 4, t);
         } else {
            fprintf(fp, "\tst.%s.b%u [%s + %u], %s;\n", space, vbits, ptr, c * vbits / 8, v);
         }
      }
      break;
   }
   case nir_intrinsic_load_ubo: {
      /* vulkan-sim reads 4 bytes per component: d0..dn, index, offset, 5 more */
      ptx_check(d->bit_size == 32, in, "load_ubo: only 32-bit components supported");
      fprintf(fp, "\tload_ubo ");
      for (unsigned c = 0; c < d->num_components; c++)
         fprintf(fp, "%s, ", reg_name(d, c));
      fprintf(fp, "%s, %s, %u, %u, %u, %u, 0;\n", src_name(&intr->src[0], 0, in),
              src_name(&intr->src[1], 0, in), nir_intrinsic_access(intr),
              nir_intrinsic_align_mul(intr), nir_intrinsic_align_offset(intr),
              nir_intrinsic_range_base(intr));
      break;
   }
   case nir_intrinsic_load_push_constant: {
      /* vulkan-sim only implements 1 or 4 destinations: pad vec2/vec3 to 4. */
      ptx_check(d->bit_size == 32, in, "load_push_constant: only 32-bit components supported");
      const char *off = src_name(&intr->src[0], 0, in);
      char offbuf[32];
      if (nir_intrinsic_base(intr) != 0) {
         const char *t = new_tmp(ctx, "b32");
         fprintf(fp, "\tadd.u32 %s, %s, %u;\n", t, off, nir_intrinsic_base(intr));
         snprintf(offbuf, sizeof(offbuf), "%s", t);
         off = offbuf;
      }
      unsigned n = d->num_components == 1 ? 1 : 4;
      char dsts[4][32];
      for (unsigned c = 0; c < n; c++) {
         if (c < d->num_components)
            snprintf(dsts[c], sizeof(dsts[c]), "%s", reg_name(d, c));
         else
            snprintf(dsts[c], sizeof(dsts[c]), "%s", new_tmp(ctx, "b32"));
      }
      fprintf(fp, "\tload_push_constant ");
      for (unsigned c = 0; c < n; c++)
         fprintf(fp, "%s, ", dsts[c]);
      fprintf(fp, "%s, 0, %u, 0, 0;\n", off, nir_intrinsic_range(intr));
      break;
   }
   case nir_intrinsic_load_vertex_id_zero_base:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_base_instance:
      ptx_check(d->num_components == 1 && d->bit_size == 32, in, "expected scalar 32-bit system value");
      fprintf(fp, "\t%s %s;\n", nir_intrinsic_infos[intr->intrinsic].name, reg_name(d, 0));
      break;
   case nir_intrinsic_vulkan_resource_index:
      ptx_check(nir_src_is_const(intr->src[0]) && nir_src_as_uint(intr->src[0]) == 0, in,
                "vulkan_resource_index: only array index 0 is supported");
      fprintf(fp, "\tload_vulkan_descriptor %s, %u, %u, %u;\n", reg_name(d, 0),
              nir_intrinsic_desc_set(intr), nir_intrinsic_binding(intr),
              nir_intrinsic_desc_type(intr));
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      fprintf(fp, "\tmov.b64 %s, %s;\n", reg_name(d, 0), src_name(&intr->src[0], 0, in));
      break;
   case nir_intrinsic_discard_if:
   case nir_intrinsic_demote_if:
   case nir_intrinsic_terminate_if:
      fprintf(fp, "\tdiscard_if %s;\n", src_name(&intr->src[0], 0, in));
      break;
   case nir_intrinsic_discard:
   case nir_intrinsic_demote:
   case nir_intrinsic_terminate: {
      const char *t = new_tmp(ctx, "u32");
      const char *p = new_tmp(ctx, "pred");
      fprintf(fp, "\tmov.u32 %s, 0;\n", t);
      fprintf(fp, "\tsetp.eq.u32 %s, %s, 0;\n", p, t);
      fprintf(fp, "\tdiscard_if %s;\n", p);
      break;
   }
   default:
      ptx_fail(in, "unsupported intrinsic %s", nir_intrinsic_infos[intr->intrinsic].name);
   }
}

/* ---- textures ----------------------------------------------------------- */

static void
emit_tex(ptx_ctx *ctx, nir_tex_instr *tex)
{
   const nir_instr *in = &tex->instr;
   ptx_check(tex->op == nir_texop_tex || tex->op == nir_texop_txb || tex->op == nir_texop_txl, in,
             "unsupported texture op %d (tex, txb, txl only)", tex->op);
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   ptx_check((tex->sampler_dim == GLSL_SAMPLER_DIM_2D || (cube && !tex->is_array)) && !tex->is_shadow, in,
             "only non-shadow 2D, 2D array and cube textures are supported");
   ptx_check(nir_alu_type_get_base_type(tex->dest_type) == nir_type_float, in,
             "only float texture results are supported");

   int coord = -1, lod = -1;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_coord:
         coord = i;
         break;
      case nir_tex_src_bias:
      case nir_tex_src_lod:
         lod = i;
         break;
      case nir_tex_src_sampler_offset:
         /* llvmpipe (lp_bld_nir.c) ignores it and samples with the base sampler; match it */
         fprintf(stderr, "nir_to_ptx: warning: dynamic sampler index ignored, as llvmpipe does\n");
         break;
      default:
         ptx_fail(in, "unsupported texture source type %d", tex->src[i].src_type);
      }
   }
   const unsigned ncoord = tex->is_array || cube ? 3 : 2;
   ptx_check(coord >= 0 && nir_src_num_components(tex->src[coord].src) == ncoord, in,
             "expected a %u-component coordinate", ncoord);

   nir_ssa_def *d = &tex->dest.ssa;
   char dsts[4][32];
   for (unsigned c = 0; c < 4; c++) {
      if (c < d->num_components)
         snprintf(dsts[c], sizeof(dsts[c]), "%s", reg_name(d, c));
      else
         snprintf(dsts[c], sizeof(dsts[c]), "%s", new_tmp(ctx, "b32"));
   }
   /* tex d0..d3, texture, sampler, s, t, layer, lod/bias, mode
    * mode = 1 bias | 2 explicit lod, +4 lod source uniform (llvmpipe then uses a
    * scalar rather than per-quad LOD), +8 array (layer valid), +16 cube (layer slot = z) */
   unsigned mode = tex->is_array ? 8 : cube ? 16 : 0;
   const char *lod_reg = "0f00000000";
   if (lod >= 0) {
      ptx_check(nir_src_bit_size(tex->src[lod].src) == 32 && nir_src_num_components(tex->src[lod].src) == 1,
                in, "expected a scalar 32-bit lod/bias");
      mode |= (tex->src[lod].src_type == nir_tex_src_bias ? 1 : 2) |
              (nir_src_is_always_uniform(tex->src[lod].src) ? 4 : 0);
      lod_reg = src_name(&tex->src[lod].src, 0, in);
   }
   fprintf(ctx->fp, "\ttex.2d.v4.f32.f32 %s, %s, %s, %s, %u, %u, %s, %s, %s, %s, %u;\n",
           dsts[0], dsts[1], dsts[2], dsts[3], tex->texture_index, tex->sampler_index,
           src_name(&tex->src[coord].src, 0, in), src_name(&tex->src[coord].src, 1, in),
           ncoord == 3 ? src_name(&tex->src[coord].src, 2, in) : "0f00000000", lod_reg, mode);
}

/* ---- control flow ------------------------------------------------------- */

static void emit_cf_list(ptx_ctx *ctx, struct exec_list *list);

/* Phis are parallel copies: predecessors write %phi_*, the phi block reads them. */
static void
emit_phi_copies_to(ptx_ctx *ctx, nir_block *pred, nir_block *succ)
{
   if (!succ)
      return;
   nir_foreach_phi(phi, succ) {
      nir_phi_src *src = NULL;
      nir_foreach_phi_src(s, phi) {
         if (s->pred == pred)
            src = s;
      }
      ptx_check(src, &phi->instr, "phi has no source for block %u", pred->index);
      nir_ssa_def *d = &phi->dest.ssa;
      for (unsigned c = 0; c < d->num_components; c++) {
         if (d->bit_size == 1)
            fprintf(ctx->fp, "\tmov.pred %s, %s;\n", phi_name(d, c), src_name(&src->src, c, &phi->instr));
         else
            fprintf(ctx->fp, "\tmov.b%u %s, %s;\n", d->bit_size, phi_name(d, c),
                    src_name(&src->src, c, &phi->instr));
      }
   }
}

static void
emit_jump(ptx_ctx *ctx, nir_jump_instr *jump)
{
   switch (jump->type) {
   case nir_jump_break:
      ptx_check(ctx->loop_depth > 0, &jump->instr, "break outside a loop");
      fprintf(ctx->fp, "\tbra loop_%u_exit;\n", ctx->loop_stack[ctx->loop_depth - 1]);
      break;
   case nir_jump_continue:
      ptx_check(ctx->loop_depth > 0, &jump->instr, "continue outside a loop");
      fprintf(ctx->fp, "\tbra loop_%u;\n", ctx->loop_stack[ctx->loop_depth - 1]);
      break;
   case nir_jump_return:
   case nir_jump_halt:
      fprintf(ctx->fp, "\tbra shader_exit;\n");
      break;
   default:
      ptx_fail(&jump->instr, "unsupported jump type %d", jump->type);
   }
}

static void
emit_block(ptx_ctx *ctx, nir_block *block)
{
   FILE *fp = ctx->fp;
   fprintf(fp, "\t// block_%u\n", block->index);

   nir_foreach_phi(phi, block) {
      nir_ssa_def *d = &phi->dest.ssa;
      for (unsigned c = 0; c < d->num_components; c++) {
         if (d->bit_size == 1)
            fprintf(fp, "\tmov.pred %s, %s;\n", reg_name(d, c), phi_name(d, c));
         else
            fprintf(fp, "\tmov.b%u %s, %s;\n", d->bit_size, reg_name(d, c), phi_name(d, c));
      }
   }

   nir_jump_instr *jump = NULL;
   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_phi:
         break;
      case nir_instr_type_alu:
         emit_alu(ctx, nir_instr_as_alu(instr));
         break;
      case nir_instr_type_deref:
         emit_deref(ctx, nir_instr_as_deref(instr));
         break;
      case nir_instr_type_intrinsic:
         emit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
         break;
      case nir_instr_type_tex:
         emit_tex(ctx, nir_instr_as_tex(instr));
         break;
      case nir_instr_type_load_const:
         emit_load_const(ctx, nir_instr_as_load_const(instr));
         break;
      case nir_instr_type_ssa_undef:
         break; /* registers are declared up front; value is undefined anyway */
      case nir_instr_type_jump:
         ptx_check(instr == nir_block_last_instr(block), instr, "jump is not the last instruction");
         jump = nir_instr_as_jump(instr);
         break;
      default:
         ptx_fail(instr, "unsupported instruction type %d", instr->type);
      }
   }

   emit_phi_copies_to(ctx, block, block->successors[0]);
   emit_phi_copies_to(ctx, block, block->successors[1]);
   if (jump)
      emit_jump(ctx, jump);
}

static void
emit_if(ptx_ctx *ctx, nir_if *nif)
{
   unsigned id = ctx->next_label++;
   ptx_check(nif->condition.is_ssa && (nif->condition.ssa->bit_size == 1 || nif->condition.ssa->bit_size == 32),
             NULL, "if condition must be a 1- or 32-bit SSA boolean");
   const char *cond = reg_name(nif->condition.ssa, 0);
   if (nif->condition.ssa->bit_size == 32) {
      const char *p = new_tmp(ctx, "pred");
      fprintf(ctx->fp, "\tsetp.ne.b32 %s, %s, 0;\n", p, cond);
      cond = p;
   }
   fprintf(ctx->fp, "\t@!%s bra if_%u_else;\n", cond, id);
   emit_cf_list(ctx, &nif->then_list);
   fprintf(ctx->fp, "\tbra if_%u_end;\n", id);
   fprintf(ctx->fp, "if_%u_else:\n", id);
   emit_cf_list(ctx, &nif->else_list);
   fprintf(ctx->fp, "if_%u_end:\n", id);
}

static void
emit_loop(ptx_ctx *ctx, nir_loop *loop)
{
   unsigned id = ctx->next_label++;
   ptx_check(ctx->loop_depth < PTX_MAX_LOOP_DEPTH, NULL, "loops nested too deeply");
   ctx->loop_stack[ctx->loop_depth++] = id;
   fprintf(ctx->fp, "loop_%u:\n", id);
   emit_cf_list(ctx, &loop->body);
   fprintf(ctx->fp, "\tbra loop_%u;\n", id);
   fprintf(ctx->fp, "loop_%u_exit:\n", id);
   ctx->loop_depth--;
}

static void
emit_cf_list(ptx_ctx *ctx, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         emit_block(ctx, nir_cf_node_as_block(node));
         break;
      case nir_cf_node_if:
         emit_if(ctx, nir_cf_node_as_if(node));
         break;
      case nir_cf_node_loop:
         emit_loop(ctx, nir_cf_node_as_loop(node));
         break;
      default:
         ptx_fail(NULL, "unsupported control-flow node %d", node->type);
      }
   }
}

/* ---- entry -------------------------------------------------------------- */

/* Puts small local arrays in registers: indirect indexing becomes selects. */
#define PTX_MAX_LOWERED_LOCAL_ARRAY 256

static void
lower_for_ptx(nir_shader *shader)
{
   NIR_PASS_V(shader, nir_lower_global_vars_to_local); /* shader_temp -> function_temp */
   NIR_PASS_V(shader, nir_lower_indirect_derefs, nir_var_function_temp, PTX_MAX_LOWERED_LOCAL_ARRAY);
   NIR_PASS_V(shader, nir_lower_vars_to_ssa);
   NIR_PASS_V(shader, nir_copy_prop);
   NIR_PASS_V(shader, nir_opt_dce);
   NIR_PASS_V(shader, nir_remove_dead_variables, nir_var_function_temp, NULL);
   /* vulkan-sim's .local loads return 0, so a local left in memory would read zeros */
   nir_foreach_function_temp_variable(var, nir_shader_get_entrypoint(shader))
      ptx_fail(NULL, "local %s could not be kept in registers (array > %u elements?); "
               "vulkan-sim's .local memory is unreliable", var->name ? var->name : "?",
               PTX_MAX_LOWERED_LOCAL_ARRAY);
   nir_foreach_variable_with_modes(var, shader, nir_var_shader_temp)
      ptx_fail(NULL, "shader_temp %s left after lowering to locals", var->name ? var->name : "?");
}

void
nir_to_ptx(nir_shader *orig, FILE *fp, unsigned shader_id)
{
   nir_shader *shader = nir_shader_clone(NULL, orig);
   lower_for_ptx(shader);
   const char *stage_name;
   switch (shader->info.stage) {
   case MESA_SHADER_VERTEX:   stage_name = "MESA_SHADER_VERTEX";   break;
   case MESA_SHADER_FRAGMENT: stage_name = "MESA_SHADER_FRAGMENT"; break;
   default: ptx_fail(NULL, "unsupported shader stage %d", shader->info.stage);
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   ptx_check(impl, NULL, "shader has no entrypoint");
   ptx_check(exec_list_length(&shader->functions) == 1, NULL,
             "expected a single (inlined) function, got %u", exec_list_length(&shader->functions));
   ptx_check(exec_list_is_empty(&impl->registers), NULL, "expected SSA-form NIR (impl has registers)");

   nir_index_ssa_defs(impl);
   nir_index_blocks(impl);

   ptx_ctx ctx = { .fp = fp, .shader = shader };
   ctx.mem_ctx = ralloc_context(NULL);
   ctx.var_names = _mesa_pointer_hash_table_create(ctx.mem_ctx);
   ctx.used_names = _mesa_set_create(ctx.mem_ctx, _mesa_hash_string, _mesa_key_string_equal);
   ctx.derefed_vars = _mesa_pointer_set_create(ctx.mem_ctx);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_deref &&
             nir_instr_as_deref(instr)->deref_type == nir_deref_type_var)
            _mesa_set_add(ctx.derefed_vars, nir_instr_as_deref(instr)->var);
      }
   }

   fprintf(fp, ".version 2.0\n");
   fprintf(fp, ".target sm_10, map_f64_to_f32\n\n");
   fprintf(fp, "// shader: %s, generated by nir_to_ptx\n", stage_name);
   /* vulkan-sim derives this name from the file's shader id. */
   fprintf(fp, ".entry %s_func%u_main () {\n", stage_name, shader_id);

   nir_foreach_variable_in_shader(var, shader)
      emit_var_decl(&ctx, var);
   nir_foreach_function_temp_variable(var, impl)
      emit_var_decl(&ctx, var);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         nir_foreach_ssa_def(instr, decl_def_cb, &ctx);
         if (instr->type == nir_instr_type_phi)
            decl_def_regs(&ctx, &nir_instr_as_phi(instr)->dest.ssa, phi_name);
      }
   }
   fprintf(fp, "\n");

   emit_cf_list(&ctx, &impl->body);

   fprintf(fp, "shader_exit:\n");
   fprintf(fp, "\tret;\n");
   fprintf(fp, "}\n");
   fflush(fp);

   ralloc_free(ctx.mem_ctx);
   ralloc_free(shader);
}
