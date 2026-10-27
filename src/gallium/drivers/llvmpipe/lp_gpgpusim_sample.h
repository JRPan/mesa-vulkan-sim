#ifndef LP_GPGPUSIM_SAMPLE_H
#define LP_GPGPUSIM_SAMPLE_H

struct llvmpipe_context;

/* Registers llvmpipe's sampler (JIT'd per FS variant/texture/sampler) as
 * vulkan-sim's texture callback for the current draw. Fragment feed only. */
void lp_gpgpusim_sample_begin(struct llvmpipe_context *lp);
/* Free JIT'd functions built for a variant / context about to be destroyed. */
struct lp_fragment_shader_variant;
void lp_gpgpusim_sample_forget_variant(const struct lp_fragment_shader_variant *variant);
void lp_gpgpusim_sample_forget_context(const struct llvmpipe_context *lp);

#endif
