#ifndef LP_GPGPUSIM_VERIFY_H
#define LP_GPGPUSIM_VERIFY_H

#include <stdbool.h>
#include <stdint.h>
#include "util/format/u_formats.h"

struct llvmpipe_context;

/* GPGPUSIM_VERIFY=1: compare each vulkan-sim draw against llvmpipe's render. */
bool lp_gpgpusim_verify_enabled(void);
/* Before gpgpusim_vkCmdDraw(): seed vulkan-sim with llvmpipe's pre-draw state. */
void lp_gpgpusim_verify_pre_draw(struct llvmpipe_context *lp);
/* After llvmpipe's own draw: compare and write metrics/images. */
void lp_gpgpusim_verify_post_draw(struct llvmpipe_context *lp);

/* Packs float RGBA the way llvmpipe's FS output stage stores it (sRGB encode
 * as lp_build_linear_to_srgb; other formats via util_format). */
void lp_gpgpusim_pack_like_llvmpipe(enum pipe_format fmt, uint8_t *dst, const float *src, unsigned w);

#endif
