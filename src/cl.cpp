#include <cassert>
#include <mutex>
#include <string>
#include <memory>
#include <utility>

#include <nlohmann/json.hpp>
#include <inja/inja.hpp>

#include "cl.h"
#include "config.h"

using json = nlohmann::json;

static inline
int round_up(int numToRound, int multiple)
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

const char *kern_basic =
#include "kernels/basic.cl"
;

const char *kern_tiled =
#include "kernels/tiled.cl"
;

template<typename T, typename... Ts>
std::string
libysmm_query_string(const T& fn, Ts... args)
{
    // Query the size
    size_t sz;
    if (cl_int err = fn(args..., 0, nullptr, &sz); err < 0)
        throw err;

    // Allocate storage
    char *temp = new char[sz];
    if (cl_int err = fn(args..., sz, temp, nullptr); err < 0)
        throw err;

    // Construct a string
    std::string ret(temp);
    delete[] temp;

    return ret;
}

struct libysmm_cl_platform
{
    libysmm_cl_platform(cl_platform_id platform);

    cl_platform_id plat_id;
    std::string name;
    std::string extensions;
};

libysmm_cl_platform::libysmm_cl_platform(cl_platform_id platform)
    : plat_id(platform)
{
    // Query the name
    name = libysmm_query_string(clGetPlatformInfo, platform, CL_PLATFORM_NAME);

    // Query the extensions
    extensions = libysmm_query_string(clGetPlatformInfo, platform,
                                      CL_PLATFORM_EXTENSIONS);
}

struct libysmm_cl_device_properties
{
    libysmm_cl_device_properties(cl_device_id dev);
    ~libysmm_cl_device_properties();

    cl_device_id dev_id;
    libysmm_cl_platform *platform;
    std::string name;
    std::string extensions;
    bool has_dp;
};

libysmm_cl_device_properties::libysmm_cl_device_properties(cl_device_id dev)
    : dev_id(dev)
{
    // Query the platform
    cl_platform_id plat_id;
    cl_int err = clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(plat_id),
                                 &plat_id, nullptr);
    if (err < 0)
        throw err;

    platform = new libysmm_cl_platform(plat_id);

    // Query the name
    name = libysmm_query_string(clGetDeviceInfo, dev, CL_DEVICE_NAME);

    // Query the extensions
    extensions = libysmm_query_string(clGetDeviceInfo, dev,
                                      CL_DEVICE_EXTENSIONS);

    // See if we have double precision support
    has_dp = extensions.find("cl_khr_fp64") != extensions.npos;
}

libysmm_cl_device_properties::~libysmm_cl_device_properties()
{
    delete platform;
}

struct libysmm_cl_handle
{
    libysmm_cl_handle(cl_context ctx, cl_device_id dev, int flags);

    std::string
    serialize() const;

    void
    unserialize(const std::string &state, int flags);

    libysmm_cl_smm_kernel *
    smm_kernel(const libysmm_smm_t *smm, double timeout);

    cl_context ctx_;
    libysmm_cl_device_properties dev_props_;
    mutable std::mutex lock_;
};

struct libysmm_cl_smm_kernel
{
    ~libysmm_cl_smm_kernel();
    libysmm_cl_smm_kernel *clone();

    cl_int
    bind(cl_mem b, cl_mem c);

    cl_int
    enqueue(cl_command_queue queue,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

    libysmm_cl_handle *h_;
    libysmm_smm_t smm_;

    cl_kernel kernel_;
    cl_mem a_;

    cl_uint work_dim_;
    size_t gs_[3];
    size_t ls_[3];
};

libysmm_cl_handle::libysmm_cl_handle(
    cl_context ctx,
    cl_device_id dev,
    int flags)
    : ctx_(ctx)
    , dev_props_(dev)
{
    assert(0 == flags);
}

std::string
libysmm_cl_handle::serialize() const
{
    std::scoped_lock guard(lock_);

    return "";
}

void
libysmm_cl_handle::unserialize(
    const std::string &state,
    int)
{
    std::scoped_lock guard(lock_);

    if ("" != state)
        throw CL_INVALID_VALUE;
}

libysmm_cl_smm_kernel *
libysmm_cl_smm_kernel::clone()
{
    auto newk = new libysmm_cl_smm_kernel(*this);

    if (a_)
        clRetainMemObject(a_);

    if (kernel_)
    {
        cl_int err;
        newk->kernel_ = clCloneKernel(kernel_, &err);

        if (err < 0)
        {
            delete newk;
            throw err;
        }
    }

    return newk;
}

libysmm_cl_smm_kernel *
libysmm_cl_handle::smm_kernel(
    const libysmm_smm_t *smm,
    double)
{
    std::scoped_lock guard(lock_);

    libysmm_dtype_t dtype = smm->dtype;
    libysmm_layout_t layout = smm->layout;
    libysmm_transpose_t transpose = smm->transpose;

    int m = smm->m, n = smm->n, k = smm->k;
    int lda = smm->lda, ldb = smm->ldb, ldc = smm->ldc;

    double alpha = smm->alpha, beta = smm->beta;

    // Validate the shape
    if (m <= 0 || n <= 0 || k <= 0)
        throw CL_INVALID_VALUE;

    // Ensure the dimensions are valid for our tiled kernel
    if (n % 32)
        throw CL_INVALID_VALUE;

    // Enusre beta is valid
    // TODO: Jason add support for beta = 1
    if (0 != beta)
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

    // Validate the A pointer
    if (nullptr == smm->a)
        throw CL_INVALID_VALUE;

    auto smmk = std::make_unique<libysmm_cl_smm_kernel>();
    smmk->h_ = this;
    smmk->smm_ = *smm;
    smmk->smm_.a = nullptr;

    /*
     * Tile A.  Each tile is 8 by 4 with the tiles being packed next
     * to each other in memory in a row-major order.  The contents of each
     * tile are stored column-major.  Here, we also handle alpha.
     */
    const int trows = 8, tcols = 4;
    const int tlda = round_up(k, tcols);
    const int tm = round_up(m, trows);
    std::vector<float> ta(tlda*tm, 0.0f);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
        {
            int tr = i / trows, trr = i % trows;
            int tc = j / tcols, tcc = j % tcols;
            int idx = tr*trows*tlda + tc*trows*tcols + tcc*trows + trr;
            ta[idx] = alpha*static_cast<float *>(smm->a)[i*lda + j];
        }

    // Copy A
    cl_int err;
    smmk->a_ = clCreateBuffer(ctx_, CL_MEM_COPY_HOST_PTR,
                              sizeof(float)*ta.size(), ta.data(), &err);
    if (err < 0)
        throw err;

    // Render the kernel
    json tplargs = {
        {"k_mod_4", k % 4}, {"m_mod_16", m % 16}
    };

    std::string ksrc = inja::render(kern_tiled, tplargs);
    const char *ksrcp = ksrc.c_str();

    // Build the program
    auto prg = clCreateProgramWithSource(ctx_, 1, &ksrcp, nullptr, &err);
    if (err < 0)
        throw err;

    err = clBuildProgram(prg, 1, &dev_props_.dev_id, nullptr, nullptr, nullptr);
    if (err < 0)
    {
        clReleaseProgram(prg);
        throw err;
    }

    // Create the kernel
    smmk->kernel_ = clCreateKernel(prg, "mm", &err);

    // Release the program, irrespective of if we created the kernel or not
    clReleaseProgram(prg);

    // See if we created the kernel
    if (err < 0)
        throw err;

    // Bind the static arguments
    err = clSetKernelArg(smmk->kernel_, 0, sizeof(smmk->a_), &smmk->a_);
    if (err < 0)
        throw err;

    const int sargs[] = { m, n, k, tlda, ldb, ldc };
    for (int i = 0; i < 6; i++)
    {
        err = clSetKernelArg(smmk->kernel_, i + 3, sizeof(int), &sargs[i]);
        if (err < 0)
            throw err;
    }

    smmk->work_dim_ = 2;

    // Columns and rows per OpenCL thread (fixed)
    const int cpt = 4;
    const int rpt = 16;

    // Blocking factors (adjustable, factor of 8 hardcoded from sub group size)
    const int blk_c = 2*8;
    const int blk_r = 1;

    smmk->ls_[0] = blk_c;
    smmk->ls_[1] = blk_r;

    smmk->gs_[0] = round_up(n, cpt*blk_c) / cpt;
    smmk->gs_[1] = round_up(m, rpt*blk_r) / rpt;

    return smmk.release();
}

libysmm_cl_smm_kernel::~libysmm_cl_smm_kernel()
{
    if (a_)
        clReleaseMemObject(a_);
    if (kernel_)
        clReleaseKernel(kernel_);
}

cl_int
libysmm_cl_smm_kernel::bind(
    cl_mem b,
    cl_mem c)
{
    if (auto err = clSetKernelArg(kernel_, 1, sizeof(b), &b); err < 0)
        return err;

    if (auto err = clSetKernelArg(kernel_, 2, sizeof(c), &c); err < 0)
        return err;

    return CL_SUCCESS;
}


cl_int
libysmm_cl_smm_kernel::enqueue(
    cl_command_queue queue,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    return clEnqueueNDRangeKernel(queue, kernel_, work_dim_, nullptr, gs_, ls_,
                                  num_events_in_wait_list, event_wait_list,
                                  event);
}

libysmm_support_level_t
libysmm_cl_get_support_level(cl_device_id id)
{
    return LIBYSMM_SUPPORT_LEVEL_BASIC;
}

cl_int
libysmm_cl_create_handle(
    libysmm_cl_handle_t *h,
    cl_context ctx,
    cl_device_id dev,
    int flags)
{
    assert(nullptr != h);

    try
    {
        *h = new libysmm_cl_handle(ctx, dev, flags);
    }
    catch (cl_int err)
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
libysmm_cl_serialize(
    libysmm_cl_handle_t h,
    char *buf,
    size_t *nbytes)
{
    assert(nullptr != h);

    if (nullptr == nbytes || (nullptr == buf && *nbytes))
        return CL_INVALID_VALUE;

    size_t nb = *nbytes;

    try
    {
        const std::string state = h->serialize();

        *nbytes = state.length();

        if (*nbytes <= nb)
        {
            state.copy(buf, *nbytes);
            return CL_SUCCESS;
        }
        else
        {
            return CL_OUT_OF_HOST_MEMORY;
        }
    }
    catch (cl_int err)
    {
        return err;
    }
}

cl_int
libysmm_cl_unserialize(
    libysmm_cl_handle_t h,
    char *state,
    size_t nbytes,
    int flags)
{
    assert(nullptr != h);

    try
    {
        h->unserialize(std::string(state, nbytes), flags);
    }
    catch (cl_int err)
    {
        return err;
    }

    return CL_SUCCESS;
}

cl_int
libysmm_cl_create_smm_kernel(
    libysmm_cl_smm_kernel_t *smmk,
    libysmm_cl_handle_t h,
    const libysmm_smm_t *smm,
    int sizeof_smm,
    double timeout
)
{
    assert(nullptr != h);
    assert(nullptr != smm);
    assert(sizeof(*smm) == sizeof_smm);

    try
    {
        *smmk = h->smm_kernel(smm, timeout);
    }
    catch (cl_int err)
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
libysmm_cl_bind_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    cl_mem b,
    cl_mem c)
{
    assert(nullptr != smmk);

    return smmk->bind(b, c);
}

cl_int
libysmm_cl_enqueue_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    cl_command_queue queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    assert(nullptr != smmk);

    return smmk->enqueue(queue, num_events_in_wait_list, event_wait_list,
                         event);
}

cl_int
libysmm_cl_clone_smm_kernel(
    libysmm_cl_smm_kernel_t smmk,
    libysmm_cl_smm_kernel_t *nsmmk
)
{
    assert(nullptr != smmk);
    assert(nullptr != nsmmk);

    try
    {
        *nsmmk = smmk->clone();
    }
    catch (cl_int err)
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
libysmm_cl_get_version(
    int *major,
    int *minor,
    int *patch,
    const char **vstr)
{
    if (major)
        *major = LIBYSMM_VERSION_MAJOR;

    if (minor)
        *minor = LIBYSMM_VERSION_MINOR;

    if (patch)
        *patch = LIBYSMM_VERSION_PATCH;

    if (vstr)
        *vstr = LIBYSMM_VERSION;
}
