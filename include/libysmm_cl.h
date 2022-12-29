#ifndef LIBYSMM_H
#define LIBYSMM_H

#include "libysmm_cl_common.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct libysmm_cl_handle *libysmm_cl_handle_t;
typedef struct libysmm_cl_smm_kernel *libysmm_cl_smm_kernel_t;

/**
 * Returns the support level for an OpenCL device.
 */
libysmm_support_level_t
libysmm_cl_get_support_level(
    cl_device_id id
);

/**
 * Creates a library handle.  Flags must be zero.
 */
cl_int
libysmm_cl_create_handle(
    libysmm_cl_handle_t *h,
    cl_context ctx,
    cl_device_id dev,
    int flags
);

/**
 * Destroys a handle.
 */
void
libysmm_cl_destroy_handle(
    libysmm_cl_handle_t h
);

/**
 * Obtains the internal state.
 */
cl_int
libysmm_cl_serialize(
    libysmm_cl_handle_t h,
    char *buf,
    size_t *nbytes
);

/**
 *
 */
cl_int
libysmm_cl_unserialize(
    libysmm_cl_handle_t h,
    char *state,
    size_t *nbytes,
    int flags
);

/**
 * Creates a new small matrix multiplication kernel.
 */
cl_int
libysmm_cl_create_smm_kernel(
    libysmm_cl_smm_kernel_t *smmk,
    libysmm_cl_handle_t h,
    const libysmm_smm_t *smm,
    int sizeof_smm,
    double timeout
);

/**
 * Destroys a kernel.
 */
void
libysmm_cl_destory_smm_kernel(
    libysmm_cl_smm_kernel_t smmk
);

/**
 * Binds an kernels arguments.
 */
cl_int
libysmm_cl_bind_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    cl_mem b,
    cl_mem c
);

/**
 * Executes a kernel.
 */
cl_int
libysmm_cl_enqueue_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    cl_command_queue queue,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event
);

cl_int
libysmm_cl_clone_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    libysmm_cl_smm_kernel_t *nsmmk
);

void
libysmm_cl_get_version(
    int *major,
    int *minor,
    int *patch,
    const char **vstr
);

#ifdef __cplusplus
}
#endif
#endif
