#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <inja/inja.hpp>

#include "libysmm_cl.h"
#include "libysmm_cl_config.h"

const char *kern_basic =
#include "kernels/basic.cl"
;

const char *kern_tiled =
#include "kernels/tiled.cl"
;

using json = nlohmann::json;

static inline
int
libysmm_round_up(int numToRound, int multiple)
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

template<typename T, typename... Ts>
static inline
std::string
libysmm_query_string(const T& fn, Ts... args)
{
    // Query the size
    size_t sz;
    if (cl_int err = fn(args..., 0, nullptr, &sz); err < 0)
        throw err;

    // Allocate storage
    auto temp = std::vector<char>(sz);
    if (cl_int err = fn(args..., sz, temp.data(), nullptr); err < 0)
        throw err;

    // Construct a string
    std::string ret(temp.data());

    return ret;
}

static inline
double
libysmm_event_profiling_info(cl_event event, cl_profiling_info param)
{
    cl_ulong t;

    cl_int err = clGetEventProfilingInfo(event, param, sizeof(t), &t, nullptr);
    if (err < 0)
        throw err;

    return t / 1e9;
}

static inline
double
libysmm_event_profiling_dt(cl_event start, cl_event end)
{
    return libysmm_event_profiling_info(end, CL_PROFILING_COMMAND_END) -
           libysmm_event_profiling_info(start, CL_PROFILING_COMMAND_START);
}

class libysmm_timer
{
public:
    libysmm_timer() : start_(now())
    {}

    double elapsed() const
    { return now() - start_; }

    void reset()
    { start_ = now(); }

private:
    static double now();

    double start_;
};

inline double
libysmm_timer::now()
{
    using namespace std::chrono;

    auto t = high_resolution_clock::now().time_since_epoch();
    auto d = duration_cast<duration<double>>(t);

    return d.count();
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
    bool has_intel_subgroups;
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

    auto has_ext = [&](const char *name)
    {
        return extensions.find(name) != extensions.npos;
    };

    // Query common extensions
    has_dp = has_ext("cl_khr_fp64");
    has_intel_subgroups = (has_ext("cl_intel_subgroups") &&
                           has_ext("cl_intel_required_subgroup_size"));
}

libysmm_cl_device_properties::~libysmm_cl_device_properties()
{
    delete platform;
}

struct libysmm_cl_handle
{
    libysmm_cl_handle(cl_context ctx, cl_device_id dev, int flags);
    ~libysmm_cl_handle();

    std::string
    serialize() const;

    void
    unserialize(const std::string &state, int flags);

    libysmm_cl_smm_kernel *
    smm_kernel(const libysmm_smm_t *smm, double timeout);

    cl_program
    build_program(const std::string &tpl, const json &args);

    cl_context ctx_;
    libysmm_cl_device_properties dev_props_;
    std::map<std::pair<std::string, json>, cl_program> prog_cache_;
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
            const cl_event *event_wait_list,
            cl_event *event);

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

libysmm_cl_handle::~libysmm_cl_handle()
{
    for (const auto &kv : prog_cache_)
        clReleaseProgram(kv.second);
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

cl_program
libysmm_cl_handle::build_program(
    const std::string &tpl,
    const json &tplargs)
{
    // Check the cache
    if (auto it = prog_cache_.find({tpl, tplargs}); it != prog_cache_.end())
        return it->second;

    std::string ksrc = inja::render(tpl, tplargs);
    const char *ksrcp = ksrc.c_str();

    // Create the program
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx_, 1, &ksrcp, nullptr, &err);
    if (err < 0)
        throw err;

    // Build the program
    err = clBuildProgram(prog, 1, &dev_props_.dev_id, nullptr, nullptr,
                         nullptr);
    if (err < 0)
    {
        clReleaseProgram(prog);
        throw err;
    }

    // Insert it into the cache and return
    prog_cache_[{tpl, tplargs}] = prog;
    return prog;
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
    const int tlda = libysmm_round_up(k, tcols);
    const int tm = libysmm_round_up(m, trows);
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
    const json tplargs = {
        {"beta", beta}, {"k_mod_4", k % 4}, {"m_mod_16", m % 16}
    };

    cl_program prog = build_program(kern_tiled, tplargs);

    // Create the kernel (should not fail)
    smmk->kernel_ = clCreateKernel(prog, "mm", &err);
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
    const int blk_c = 1*8;
    const int blk_r = 1;

    smmk->ls_[0] = blk_c;
    smmk->ls_[1] = blk_r;

    smmk->gs_[0] = libysmm_round_up(n, cpt*blk_c) / cpt;
    smmk->gs_[1] = libysmm_round_up(m, rpt*blk_r) / rpt;

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
    try
    {
        libysmm_cl_device_properties props(id);

        if (props.has_intel_subgroups)
            return LIBYSMM_SUPPORT_LEVEL_TUNED;
        else
            return LIBYSMM_SUPPORT_LEVEL_BASIC;
    }
    catch (...)
    {
        return LIBYSMM_SUPPORT_LEVEL_NONE;
    }
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
    catch (const std::bad_alloc &)
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
    catch (const std::bad_alloc &)
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
    catch (const std::bad_alloc &)
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
