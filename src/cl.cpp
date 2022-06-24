#include <cassert>
#include <mutex>
#include <memory>
#include <new>
#include <utility>

#include "cl.h"

struct libysmm_cl_handle
{
    libysmm_cl_handle(cl_context ctx, cl_device_id dev)
        : ctx_(ctx)
        , dev_(dev)
    {}

    libysmm_cl_smm_kernel *
    smm_kernel(const libysmm_smm_t *smm);

    cl_context ctx_;
    cl_device_id dev_;
    std::mutex lock_;
};

struct libysmm_cl_smm_kernel
{
    ~libysmm_cl_smm_kernel();

    cl_int
    enqueue(cl_mem a, cl_mem b, cl_mem c,
            cl_command_queue queue,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

    libysmm_cl_handle *h_;
    libysmm_smm_t smm_;

    cl_kernel kernel_;
    cl_uint work_dim_;
    size_t gs_[3];
    size_t ls_[3];
};

libysmm_cl_smm_kernel *
libysmm_cl_handle::smm_kernel(const libysmm_smm_t *smm)
{
    libysmm_dtype_t dtype = smm->dtype;
    libysmm_layout_t layout = smm->layout;
    libysmm_transpose_t transpose = smm->transpose;

    int m = smm->m, n = smm->n, k = smm->k;
    int lda = smm->lda, ldb = smm->ldb, ldc = smm->ldc;

    // Validate the shape
    if (m <= 0 || n <= 0 || k <= 0)
        throw CL_INVALID_VALUE;

    // Validate the data type
    if (LIBYSMM_DTYPE_FP32 != dtype)
        throw CL_INVALID_VALUE;

    // Validate the transpose
    if (LIBYSMM_TRANSPOSE_NN != transpose)
        throw CL_INVALID_VALUE;

    // Validate the layout
    if (LIBYSMM_LAYOUT_ROW_MAJOR == layout)
    {
        if (k > lda || n > ldb || n > ldc)
            throw CL_INVALID_VALUE;
    }
    else
    {
        throw CL_INVALID_VALUE;
    }

    auto smmk = new libysmm_cl_smm_kernel();
    smmk->h_ = this;
    smmk->smm_ = *smm;

    const char *ktpl = R"""(
        __kernel void
        mm(__global const float* restrict a,
           __global const float* restrict b,
           __global float* restrict c)
        {
            const int M = %d, N = %d, K = %d, lda = %d, ldb = %d, ldc = %d;
            const int rx = get_global_id(0);
            const int cx = get_global_id(1);

            if (rx < M && cx < N)
            {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += a[rx*lda + k] * b[k*ldb + cx];

                c[rx*ldc + cx] = acc;
            }
        }
    )""";

    int ksz = snprintf(nullptr, 0, ktpl, m, n, k, lda, ldb, ldc);
    auto ksrc = std::make_unique<char[]>(ksz + 1);
    snprintf(ksrc.get(), ksz + 1, ktpl, m, n, k, lda, ldb, ldc);

    puts(ksrc.get());

    cl_int err;
    const char *ksrcp = ksrc.get();
    auto prg = clCreateProgramWithSource(ctx_, 1, &ksrcp, nullptr, &err);
    if (err < 0)
        throw err;

    err = clBuildProgram(prg, 1, &dev_, nullptr, nullptr, nullptr);
    if (err < 0)
    {
        clReleaseProgram(prg);
        throw err;
    }

    smmk->kernel_ = clCreateKernel(prg, "mm", &err);
    if (err < 0)
    {
        clReleaseProgram(prg);
        throw err;
    }

    smmk->work_dim_ = 2;
    smmk->ls_[0] = smmk->ls_[1] = 16;
    smmk->gs_[0] = ((m + smmk->ls_[0] - 1) / smmk->ls_[0])*smmk->ls_[0];
    smmk->gs_[1] = ((n + smmk->ls_[1] - 1) / smmk->ls_[1])*smmk->ls_[1];

    printf("%p\n", smmk->kernel_);

    return smmk;
}

libysmm_cl_smm_kernel::~libysmm_cl_smm_kernel()
{
    if (kernel_)
        clReleaseKernel(kernel_);
}

cl_int
libysmm_cl_smm_kernel::enqueue(
    cl_mem a,
    cl_mem b,
    cl_mem c,
    cl_command_queue queue,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    if (auto err = clSetKernelArg(kernel_, 0, sizeof(a), &a); err < 0)
        return err;

    if (auto err = clSetKernelArg(kernel_, 1, sizeof(b), &b); err < 0)
        return err;

    if (auto err = clSetKernelArg(kernel_, 2, sizeof(c), &c); err < 0)
        return err;

    return clEnqueueNDRangeKernel(queue, kernel_, work_dim_, nullptr, gs_, ls_,
                                  num_events_in_wait_list, event_wait_list,
                                  event);
}

cl_int
libysmm_cl_create_handle(
    libysmm_cl_handle_t *h,
    cl_context ctx,
    cl_device_id dev,
    int flags)
{
    assert(0 == flags);
    assert(nullptr != h);

    try
    {
        *h = new libysmm_cl_handle(ctx, dev);
    }
    catch (int err)
    {
        return err;
    }
    catch (const std::bad_alloc&)
    {
        return CL_OUT_OF_HOST_MEMORY;
    }

    return CL_SUCCESS;
}

void
libysmm_cl_destroy_handle(
    libysmm_cl_handle_t h)
{
    assert(nullptr != h);

    delete h;
}

cl_int
libysmm_cl_create_smm_kernel(
    libysmm_cl_smm_kernel_t *smmk,
    libysmm_cl_handle_t h,
    const libysmm_smm_t *smm,
    int sizeof_smm
)
{
    assert(nullptr != h);
    assert(nullptr != smm);
    assert(sizeof(*smm) == sizeof_smm);

    try
    {
        *smmk = h->smm_kernel(smm);
    }
    catch (int err)
    {
        return err;
    }
    catch (const std::bad_alloc&)
    {
        return CL_OUT_OF_HOST_MEMORY;
    }

    return CL_SUCCESS;
}

void
libysmm_cl_destory_smm_kernel(
    libysmm_cl_smm_kernel_t smmk)
{
    assert(nullptr != smmk);

    delete smmk;
}

cl_int
libysmm_cl_enqueue_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    cl_mem a,
    cl_mem b,
    cl_mem c,
    cl_command_queue queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    assert(nullptr != smmk);

    return smmk->enqueue(a, b, c, queue, num_events_in_wait_list,
                         event_wait_list, event);
}
