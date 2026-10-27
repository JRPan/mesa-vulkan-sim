#ifndef LP_GPGPUSIM_FEED_H
#define LP_GPGPUSIM_FEED_H

#include <stdbool.h>

struct llvmpipe_context;
struct lp_rasterizer_task;
struct lp_rast_shader_inputs;

/* GPGPUSIM_FRAG_FEED=1: vulkan-sim's PS runs on llvmpipe's fragments. */
bool lp_gpgpusim_feed_enabled(void);
/* Start recording (after the pre-draw flush); fragments of every flush until _end belong to this draw. */
void lp_gpgpusim_feed_begin(struct llvmpipe_context *lp);
/* Called by the rasterizer right before each FS JIT call (4x4 block, linear mask bit = y*4 + x). */
void lp_gpgpusim_feed_record(struct lp_rasterizer_task *task,
                             const struct lp_rast_shader_inputs *inputs,
                             unsigned x, unsigned y, unsigned mask);
/* Called on FS paths the feed cannot observe; aborts while recording. */
void lp_gpgpusim_feed_check_unsupported_path(const char *path);
/* Stop recording (after the post-draw flush) and order fragments as llvmpipe shaded them. */
void lp_gpgpusim_feed_end(void);

/* State the rasterizer shaded this draw with (NULL if no fragments); valid after _end. */
struct lp_rast_state;
const struct lp_rast_state *lp_gpgpusim_feed_state(void);

unsigned lp_gpgpusim_feed_count(void);
unsigned lp_gpgpusim_feed_num_inputs(void);
/* Records come in whole 2x2 quads (count % 4 == 0), lanes TL, TR, BL, BR.
 * helper: uncovered (pixel == UINT32_MAX). inputs: num_inputs x 4 floats,
 * indexed by FS input slot (driver_location). */
void lp_gpgpusim_feed_get(unsigned i, unsigned *pixel, float *z, bool *helper, const float **inputs);

#endif
