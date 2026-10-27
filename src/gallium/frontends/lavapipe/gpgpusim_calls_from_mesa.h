// Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef GPGPUSIM_CALLS_FROM_MESA_H
#define GPGPUSIM_CALLS_FROM_MESA_H

/* Set by lavapipe on swapchain images' pipe_resource; lets llvmpipe tell final
 * output from render-to-texture (GPGPUSIM_SKIP_RTT). */
#define LVP_RESOURCE_FLAG_SWAPCHAIN PIPE_RESOURCE_FLAG_FRONTEND_PRIV

#include "vulkan/vulkan_core.h"
#include "lvp_acceleration_structure.h"

// #include "anv_private.h"

//extern void gpgpusim_init();
extern void gpgpusim_setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
extern void gpgpusim_setGeometries(const VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
extern void gpgpusim_addTreelets(VkAccelerationStructureKHR accelerationStructure);
extern void gpgpusim_testTraversal(struct lvp_bvh_node* root);
extern uint32_t gpgpusim_registerShader(char * shaderPath, uint32_t shader_type);
extern void gpgpusim_allocBLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr);
extern void gpgpusim_allocTLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr);
extern void* gpgpusim_allocBuffer(void* bufferAddr, uint64_t bufferSize);

extern void gpgpusim_vkCmdTraceRaysKHR(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);

extern void gpgpusim_setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
extern void gpgpusim_setDescriptorSet(unsigned index, struct lvp_descriptor_set *set);

// For trace runner
extern void gpgpusim_addTreelets_cpp(VkAccelerationStructureKHR accelerationStructure);
extern uint32_t gpgpusim_registerShader_cpp(char * shaderPath, uint32_t shader_type);

extern void gpgpusim_vkCmdTraceRaysKHR_cpp(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);

extern void gpgpusim_setDescriptorSet_cpp(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);

extern void gpgpusim_pass_child_addr(void* address);

extern void gpgpusim_vkCmdDraw();
extern void gpgpusim_bindVertex(unsigned index, float *addr, unsigned size, unsigned stride);
extern void gpgpusim_saveIndexBuffer(void *ptr, unsigned index_size, unsigned buf_size);
extern void gpgpusim_saveVertexInfo(unsigned location, unsigned binding, unsigned offset, unsigned rate);
extern void gpgpusim_saveInstance(unsigned instanceCount, unsigned startInstance);
extern void gpgpusim_savePipeCtx(void *ctx);
extern void gpgpusim_saveUBO(unsigned stage, unsigned index, unsigned offset, unsigned size, void *addr);
extern void gpgpusim_saveIntrinsic(unsigned StartVertex, unsigned BaseVertex, unsigned instanceCount, unsigned startInstance);
extern void gpgpusim_saveViewport(float width, float height, float x, float y, unsigned depthcmpOp, float min_depth, float max_depth);
extern void gpgpusim_loadFramebuffer(unsigned width, unsigned height, const float *rgba, const float *depth);
extern void gpgpusim_getFramebuffer(unsigned *width, unsigned *height, const float **rgba, const float **depth);
extern int gpgpusim_lastDrawRendered(void);
extern int gpgpusim_fragmentsPending(void);
/* Texture sampling provided by mesa for one 2x2 quad (lanes TL, TR, BL, BR); texel[lane][chan].
 * lod_mode = 0 implicit | 1 bias | 2 explicit lod, +4 if the lod source is uniform, +8 array (r = layer), +16 cube (r = z).
 * addr/naddr: texel addresses each lane reads (timing), texel_bytes each. */
#define GPGPUSIM_MAX_TEXEL_ADDRS 8
typedef void (*gpgpusim_sample_quad_fn)(unsigned texture_index, unsigned sampler_index, unsigned lod_mode,
                                        const float s[4], const float t[4], const float r[4], const float lod[4],
                                        float texel[4][4], uint64_t addr[4][GPGPUSIM_MAX_TEXEL_ADDRS],
                                        unsigned naddr[4], unsigned *texel_bytes);
extern void gpgpusim_setSampleCallback(gpgpusim_sample_quad_fn fn);
/* Blend (+ colour mask / logic op) of one fragment into the vulkan-sim FBO pixel (float RGBA, in/out). */
typedef void (*gpgpusim_blend_pixel_fn)(float dst[4], const float src[4]);
extern void gpgpusim_setBlendCallback(gpgpusim_blend_pixel_fn fn);
extern void gpgpusim_runFragments(unsigned count, const unsigned *pixels, const float *z,
                                  const uint8_t *helper, int depth_write, unsigned num_attribs,
                                  const char *const *ids, const unsigned *ncomp,
                                  const float *const *values);
extern unsigned gpgpusim_getVSOutput(const char *identifier, unsigned instance, unsigned vertex_index, float *out);
#endif /* GPGPUSIM_CALLS_FROM_MESA_H */
