#ifndef NIR_TO_PTX_H
#define NIR_TO_PTX_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nir_shader;

/* Writes the vulkan-sim PTX for a VS/FS; the entry is named from shader_id.
 * Takes SSA-form NIR; lowers and emits a private clone, leaving it unchanged. */
void nir_to_ptx(struct nir_shader *shader, FILE *fp, unsigned shader_id);

/* vulkan-sim's identifier for an I/O variable, e.g. "VARYING_SLOT_VAR0_xyzw". */
struct nir_variable;
const char *nir_to_ptx_var_location(const struct nir_shader *shader,
                                    const struct nir_variable *var,
                                    char *buf, size_t size);

#ifdef __cplusplus
}
#endif

#endif
