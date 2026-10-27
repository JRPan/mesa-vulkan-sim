/*
 * Fragment feed (GPGPUSIM_FRAG_FEED=1): records the fragments llvmpipe's
 * rasterizer hands to its fragment shader, with inputs interpolated exactly as
 * the FS JIT does (lp_bld_interp.c attribs_update_simple), so vulkan-sim's PS
 * runs on llvmpipe's coverage and inputs. See docs/verification.md.
 */

#include "lp_gpgpusim_feed.h"

#include "lp_bld_interp.h"
#include "lp_context.h"
#include "lp_limits.h"
#include "lp_rast.h"
#include "lp_rast_priv.h"
#include "lp_scene.h"
#include "lp_state_fs.h"
#include "util/u_cpu_detect.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "gallivm/lp_bld_sample.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct feed_rec {
   uint32_t tile;     /* tile index, row-major */
   uint32_t tile_seq; /* order of the shade call within its tile */
   uint32_t pixel;    /* y * fb_width + x */
   uint32_t vals;     /* offset into the owning thread's value array */
   uint16_t thread;
   uint8_t order;     /* quad * 4 + lane within the 4x4 block (llvmpipe's lane order) */
   uint8_t helper;    /* not covered: runs only to provide quad derivatives */
   float z;
};

static struct {
   bool enabled_checked, enabled;
   volatile bool active;
   unsigned fb_w, fb_h, tiles_x;
   unsigned num_inputs;
   unsigned stride; /* floats per fragment: num_inputs * 4 */
   enum lp_interp interp[PIPE_MAX_SHADER_INPUTS];
   float pos_offset;
   uint32_t *tile_seq;
   struct util_dynarray recs[LP_MAX_THREADS];
   struct util_dynarray vals[LP_MAX_THREADS];
   /* merged result */
   struct feed_rec *sorted;
   unsigned count;
   /* the rasterizer state this draw was shaded with (variant, jit context and
    * resources); its scene-memory arrays are copied, as the scene is recycled */
   simple_mtx_t state_mtx;
   struct lp_rast_state state;
   bool state_valid;
   uint8_t u8_blend_color[4 * 16] __attribute__((aligned(16)));
   float f_blend_color[LP_MAX_VECTOR_LENGTH / 4] __attribute__((aligned(16)));
} feed;

bool
lp_gpgpusim_feed_enabled(void)
{
   if (!feed.enabled_checked) {
      feed.enabled_checked = true;
      const char *v = getenv("GPGPUSIM_FRAG_FEED");
      feed.enabled = v && strcmp(v, "0") != 0;
      for (unsigned t = 0; t < LP_MAX_THREADS; t++) {
         util_dynarray_init(&feed.recs[t], NULL);
         util_dynarray_init(&feed.vals[t], NULL);
      }
      simple_mtx_init(&feed.state_mtx, mtx_plain);
   }
   return feed.enabled;
}

void
lp_gpgpusim_feed_begin(struct llvmpipe_context *lp)
{
   const struct lp_fragment_shader *fs = lp->fs;
   if (!util_get_cpu_caps()->has_fma) {
      /* The JIT's llvm.fmuladd is only fused on FMA hosts; fmaf() below assumes that. */
      fprintf(stderr, "gpgpusim feed: host CPU lacks FMA; interpolation would not match the JIT\n");
      abort();
   }
   if (util_framebuffer_get_num_samples(&lp->framebuffer) > 1) {
      fprintf(stderr, "gpgpusim feed: multisampled framebuffers are not supported\n");
      abort();
   }
   feed.fb_w = lp->framebuffer.width;
   feed.fb_h = lp->framebuffer.height;
   feed.tiles_x = DIV_ROUND_UP(feed.fb_w, TILE_SIZE);
   unsigned tiles = feed.tiles_x * DIV_ROUND_UP(feed.fb_h, TILE_SIZE);
   FREE(feed.tile_seq);
   feed.tile_seq = CALLOC(tiles, sizeof(uint32_t));

   feed.num_inputs = fs->info.base.num_inputs;
   feed.stride = feed.num_inputs * 4;
   for (unsigned i = 0; i < feed.num_inputs; i++)
      feed.interp[i] = fs->inputs[i].interp;
   feed.pos_offset = fs->info.base.properties[TGSI_PROPERTY_FS_COORD_PIXEL_CENTER] ? 0.0f : 0.5f;

   for (unsigned t = 0; t < LP_MAX_THREADS; t++) {
      util_dynarray_clear(&feed.recs[t]);
      util_dynarray_clear(&feed.vals[t]);
   }
   FREE(feed.sorted);
   feed.sorted = NULL;
   feed.count = 0;
   feed.state_valid = false;
   feed.active = true;
}

static void
snapshot_state(const struct lp_rast_state *state)
{
   if (p_atomic_read(&feed.state_valid) && feed.state.variant == state->variant)
      return;
   simple_mtx_lock(&feed.state_mtx);
   if (!feed.state_valid) {
      feed.state = *state;
      if (state->jit_context.u8_blend_color) {
         memcpy(feed.u8_blend_color, state->jit_context.u8_blend_color, sizeof(feed.u8_blend_color));
         memcpy(feed.f_blend_color, state->jit_context.f_blend_color, sizeof(feed.f_blend_color));
      }
      feed.state.jit_context.u8_blend_color = feed.u8_blend_color;
      feed.state.jit_context.f_blend_color = feed.f_blend_color;
      feed.state.jit_resources.aniso_filter_table = lp_build_sample_aniso_filter_table();
      /* scene memory, recycled after the flush; nothing downstream may read it */
      feed.state.jit_context.viewports = NULL;
      memset(feed.state.jit_resources.constants, 0, sizeof(feed.state.jit_resources.constants));
      memset(feed.state.jit_resources.ssbos, 0, sizeof(feed.state.jit_resources.ssbos));
      p_atomic_set(&feed.state_valid, true);
   } else if (feed.state.variant != state->variant) {
      /* one draw is shaded with one state; anything else breaks the snapshot */
      fprintf(stderr, "gpgpusim feed: fragments of one draw came with different FS variants\n");
      abort();
   }
   simple_mtx_unlock(&feed.state_mtx);
}

const struct lp_rast_state *
lp_gpgpusim_feed_state(void)
{
   return feed.state_valid ? &feed.state : NULL;
}

void
lp_gpgpusim_feed_record(struct lp_rasterizer_task *task,
                        const struct lp_rast_shader_inputs *inputs,
                        unsigned x, unsigned y, unsigned mask)
{
   if (!feed.active)
      return;

   snapshot_state(task->state);
   const float (*a0)[4] = (const float (*)[4])GET_A0(inputs);
   const float (*dadx)[4] = (const float (*)[4])GET_DADX(inputs);
   const float (*dady)[4] = (const float (*)[4])GET_DADY(inputs);
   const unsigned t = task->thread_index;
   const unsigned tile = (y / TILE_SIZE) * feed.tiles_x + x / TILE_SIZE;
   const uint32_t seq = feed.tile_seq[tile]++; /* one thread owns a tile at a time */

   /* Whole 2x2 quads in the JIT's lane order (TL, TR, BL, BR); uncovered
    * pixels of a covered quad become helpers, as in the JIT. */
   for (unsigned k = 0; k < 16; k++) {
      const unsigned qx = (k / 4) % 2 * 2, qy = (k / 4) / 2 * 2;
      const unsigned quad_bits = 0x33u << (qy * 4 + qx);
      if (!(mask & quad_bits))
         continue;
      const unsigned bit = (qy + (k % 4) / 2) * 4 + qx + (k % 4) % 2;
      const unsigned px = x + bit % 4, py = y + bit / 4;
      const bool covered = (mask & (1u << bit)) && px < feed.fb_w && py < feed.fb_h;
      const float fx = (float)px, fy = (float)py;

      /* a = fmuladd(dady, y, fmuladd(dadx, x, a0)); position z gets the polygon offset in a0[0].x */
      const float w = fmaf(dady[0][3], fy, fmaf(dadx[0][3], fx, a0[0][3]));
      const float z = fmaf(dady[0][2], fy, fmaf(dadx[0][2], fx, a0[0][2])) + a0[0][0];
      const float oow = 1.0f / w;

      struct feed_rec rec = {
         .tile = tile, .tile_seq = seq,
         .pixel = covered ? py * feed.fb_w + px : UINT32_MAX,
         .vals = util_dynarray_num_elements(&feed.vals[t], float),
         .thread = t, .order = k, .helper = !covered, .z = z,
      };
      float *v = util_dynarray_grow(&feed.vals[t], float, feed.stride);
      for (unsigned i = 0; i < feed.num_inputs; i++) {
         const unsigned s = i + 1; /* setup slot 0 is position */
         for (unsigned c = 0; c < 4; c++) {
            float a;
            switch (feed.interp[i]) {
            case LP_INTERP_PERSPECTIVE:
               a = fmaf(dady[s][c], fy, fmaf(dadx[s][c], fx, a0[s][c])) * oow;
               break;
            case LP_INTERP_LINEAR:
               a = fmaf(dady[s][c], fy, fmaf(dadx[s][c], fx, a0[s][c]));
               break;
            case LP_INTERP_CONSTANT:
            case LP_INTERP_FACING:
               a = a0[s][c];
               break;
            case LP_INTERP_POSITION:
               a = c == 0 ? fx + feed.pos_offset : c == 1 ? fy + feed.pos_offset : c == 2 ? z : w;
               break;
            default:
               fprintf(stderr, "gpgpusim feed: unknown interpolation %d\n", feed.interp[i]);
               abort();
            }
            v[i * 4 + c] = a;
         }
      }
      util_dynarray_append(&feed.recs[t], struct feed_rec, rec);
   }
}

void
lp_gpgpusim_feed_check_unsupported_path(const char *path)
{
   if (feed.active) {
      fprintf(stderr, "gpgpusim feed: llvmpipe took the %s path, which bypasses the recorded "
                      "FS invocation (set LP_PERF=no_rast_linear, LP_DEBUG=no_fastpath)\n", path);
      abort();
   }
}

static int
rec_cmp(const void *pa, const void *pb)
{
   const struct feed_rec *a = pa, *b = pb;
   if (a->tile != b->tile)
      return a->tile < b->tile ? -1 : 1;
   if (a->tile_seq != b->tile_seq)
      return a->tile_seq < b->tile_seq ? -1 : 1;
   return (int)a->order - (int)b->order;
}

void
lp_gpgpusim_feed_end(void)
{
   feed.active = false;
   unsigned n = 0;
   for (unsigned t = 0; t < LP_MAX_THREADS; t++)
      n += util_dynarray_num_elements(&feed.recs[t], struct feed_rec);
   feed.sorted = MALLOC(sizeof(struct feed_rec) * MAX2(n, 1));
   feed.count = 0;
   for (unsigned t = 0; t < LP_MAX_THREADS; t++) {
      unsigned m = util_dynarray_num_elements(&feed.recs[t], struct feed_rec);
      memcpy(feed.sorted + feed.count, feed.recs[t].data, m * sizeof(struct feed_rec));
      feed.count += m;
   }
   /* tile, then rasterization order within the tile (per-pixel order matches
    * llvmpipe); quads stay as 4 consecutive records */
   qsort(feed.sorted, feed.count, sizeof(struct feed_rec), rec_cmp);
}

unsigned
lp_gpgpusim_feed_count(void)
{
   return feed.count;
}

unsigned
lp_gpgpusim_feed_num_inputs(void)
{
   return feed.num_inputs;
}

void
lp_gpgpusim_feed_get(unsigned i, unsigned *pixel, float *z, bool *helper, const float **inputs)
{
   assert(i < feed.count);
   const struct feed_rec *r = &feed.sorted[i];
   *pixel = r->pixel;
   *z = r->z;
   *helper = r->helper;
   *inputs = (const float *)feed.vals[r->thread].data + r->vals;
}
