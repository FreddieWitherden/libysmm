#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <random>
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
    return std::string(temp.data());
}

template<typename T>
std::vector<T> libysmm_random_vec(size_t n, T min=0.1, T max=1.0)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<T> dist(min, max);

    std::vector<T> mat(n);
    std::generate(mat.begin(), mat.end(), [&]() { return dist(gen); });
    return mat;
}

template<typename F>
static
auto
libysmm_tile_matrix(int m, int k, int trows, int tcols, F f)
{
    const int tlda = libysmm_round_up(k, tcols);
    const int tm = libysmm_round_up(m, trows);
    std::vector<decltype(f(0, 0))> mat(tlda*tm, 0);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            int tr = i / trows, trr = i % trows;
            int tc = j / tcols, tcc = j % tcols;
            int idx = tr*trows*tlda + tc*trows*tcols + tcc*trows + trr;
            mat[idx] = f(i, j);
        }
    }

    return std::make_tuple(mat, tm, tlda);
}

static inline
double
libysmm_event_profiling_info(cl_event event, cl_profiling_info param)
{
    cl_ulong t;

    cl_int err = clGetEventProfilingInfo(event, param, sizeof(t), &t, nullptr);
    return (err) ? -1 : t / 1e9;
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
    cl_command_queue queue_;
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

    // Attempt to create a command queue for profiling
    const cl_command_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    queue_ = clCreateCommandQueueWithProperties(ctx, dev, props, nullptr);
}

libysmm_cl_handle::~libysmm_cl_handle()
{
    if (queue_)
        clReleaseCommandQueue(queue_);

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

template<typename T>
double
libysmm_benchmark_kernel(cl_context ctx, cl_command_queue queue,
                         std::unique_ptr<libysmm_cl_smm_kernel> &kern,
                         int nbench = 50)
{
    double ret = -1;

    size_t b_sz = kern->smm_.k*kern->smm_.ldb;
    size_t c_sz = kern->smm_.m*kern->smm_.ldc;

    auto rand = libysmm_random_vec<T>(std::max(b_sz, c_sz));

    // Swap out the kernel for a clone so we can bind the arguments
    cl_kernel orig_kernel = kern->kernel_;
    kern->kernel_ = clCloneKernel(orig_kernel, nullptr);

    // Allocate some temporary buffers for the kernel to use
    cl_mem b_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  b_sz*sizeof(T), rand.data(), nullptr);
    cl_mem c_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  c_sz*sizeof(T), rand.data(), nullptr);

    if (kern->kernel_ && b_buf && c_buf)
    {
        cl_event start = nullptr, end = nullptr;

        // Configure the kernel arguments
        cl_int err = kern->bind(b_buf, c_buf);

        // Benchmark the kernel
        for (int i = 0; !err && i < nbench + 1; i++)
        {
            if (1 == i)
                err = kern->enqueue(queue, 0, nullptr, &start);
            else if (nbench == i)
                err = kern->enqueue(queue, 0, nullptr, &end);
            else
                err = kern->enqueue(queue, 0, nullptr, nullptr);
        }

        // Wait for the kernels to finish and measure the time
        if (!clFinish(queue) && !err)
            ret = libysmm_event_profiling_dt(start, end);

        if (start)
            clReleaseEvent(start);
        if (end)
            clReleaseEvent(end);
    }

    // Release cloned kernel and our buffers
    if (kern->kernel_)
        clReleaseKernel(kern->kernel_);
    if (b_buf)
        clReleaseMemObject(b_buf);
    if (c_buf)
        clReleaseMemObject(c_buf);

    // Restore the old kernel
    kern->kernel_ = orig_kernel;

    return ret;
}

libysmm_cl_smm_kernel *
libysmm_cl_handle::smm_kernel(
    const libysmm_smm_t *smm,
    double)
{
    std::scoped_lock guard(lock_);

    libysmm_dtype_t dtype = smm->dtype;

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

    // Validate the layout
    if (k > lda || n > ldb || n > ldc)
        throw CL_INVALID_VALUE;

    // Validate the A pointer
    if (nullptr == smm->a)
        throw CL_INVALID_VALUE;

    auto smmk = std::make_unique<libysmm_cl_smm_kernel>();
    smmk->h_ = this;
    smmk->smm_ = *smm;
    smmk->smm_.a = nullptr;

    // Tile the matrix
    auto a_at = [&](int i, int j) -> float
    {
        return alpha*static_cast<float *>(smm->a)[i*lda + j];
    };
    auto [ta, tm, tk] = libysmm_tile_matrix(m, k, 8, 4, a_at);

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

    const int sargs[] = { m, n, k, tk, ldb, ldc };
    for (int i = 0; CL_SUCCESS == err && i < 6; i++)
        err = clSetKernelArg(smmk->kernel_, i + 3, sizeof(int), &sargs[i]);

    if (err < 0)
        throw err;

    smmk->work_dim_ = 2;

    // Columns and rows per OpenCL thread (fixed)
    const int cpt = 4;
    const int rpt = 16;

    // Possible blocking factors
    const int blockings[][2] = {
        {1, 1}, {2, 1}, {1, 2}, {2, 2}, {2, 4}, {4, 2}, {4, 4}
    };
    int best_blk_c = blockings[0][0], best_blk_r = blockings[0][1];
    double best_dt = 0;

    // Benchmark the factors to see which one is best
    if (queue_)
    {
        for (auto [blk_c, blk_r] : blockings)
        {
            smmk->ls_[0] = 8*blk_c;
            smmk->ls_[1] = blk_r;

            smmk->gs_[0] = libysmm_round_up(n, 8*cpt*blk_c) / cpt;
            smmk->gs_[1] = libysmm_round_up(m, rpt*blk_r) / rpt;

            double dt = libysmm_benchmark_kernel<float>(ctx_, queue_, smmk);
            if (0 == best_dt || (dt > 0 && dt < best_dt))
            {
                best_blk_c = blk_c;
                best_blk_r = blk_r;
                best_dt = dt;
            }
        }
    }

    // Go with the best set of factors
    smmk->ls_[0] = 8*best_blk_c;
    smmk->ls_[1] = best_blk_r;

    smmk->gs_[0] = libysmm_round_up(n, 8*cpt*best_blk_c) / cpt;
    smmk->gs_[1] = libysmm_round_up(m, rpt*best_blk_r) / rpt;

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
