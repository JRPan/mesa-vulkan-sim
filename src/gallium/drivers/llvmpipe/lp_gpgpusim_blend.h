#ifndef LP_GPGPUSIM_BLEND_H
#define LP_GPGPUSIM_BLEND_H

struct llvmpipe_context;

/* Registers llvmpipe's blend stage (JIT'd per FS variant) as vulkan-sim's
 * per-fragment blend callback for the current draw, or NULL when the draw
 * neither blends, masks nor uses a logic op. Fragment feed only. */
void lp_gpgpusim_blend_begin(struct llvmpipe_context *lp);
/* Free JIT'd functions built for a variant / context about to be destroyed. */
struct lp_fragment_shader_variant;
void lp_gpgpusim_blend_forget_variant(const struct lp_fragment_shader_variant *variant);
void lp_gpgpusim_blend_forget_context(const struct llvmpipe_context *lp);

#endif
