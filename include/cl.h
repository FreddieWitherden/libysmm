#ifndef LIBYSMM_CL_H
#define LIBYSMM_CL_H

#include "common.h"

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
 * Creates a new small matrix multiplication kernel.
 */
cl_int
libysmm_cl_create_smm_kernel(
    libysmm_cl_smm_kernel_t *smmk,
    libysmm_cl_handle_t h,
    const libysmm_smm_t *smm,
    int sizeof_smm
);

/**
 * Destroys a kernel.
 */
void
libysmm_cl_destory_smm_kernel(
    libysmm_cl_smm_kernel_t smmk
);

/**
 * Executes a kernel.
 */
cl_int
libysmm_cl_enqueue_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    cl_command_queue queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event
);

#ifdef __cplusplus
}
#endif
#endif