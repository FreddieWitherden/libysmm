# libysmm

libysmm is a library for performing small matrix multiplications on GPUs.

## Dependencies

Beyond OpenCL, libysmm depends on the header-only libraries:
 - [Inja](https://github.com/pantor/inja)
 - [Json](https://github.com/nlohmann/json)

Both libraries are fetched automatically by Cmake using FetchContent.

## Build

```sh
git clone https://github.com/FreddieWitherden/libysmm.git
cd libysmm/
mkdir build && cd build
cmake ./../
make
```

## API

```c
cl_device_id dev;
cl_context ctx;
cl_command_queue queue;
cl_mem bufA, bufB, bufC;
int M, N, K;

// ...

libysmm_cl_handle_t h;
libysmm_cl_create_handle(&h, ctx, dev, 0);

libysmm_smm_t smm = {
    .dtype = LIBYSMM_DTYPE_FP32,
    .layout = LIBYSMM_LAYOUT_ROW_MAJOR,
    .transpose = LIBYSMM_TRANSPOSE_NN,
    .m = M, .n = N, .k = K,
    .lda = K, .ldb = N, .ldc = N,
    .alpha = 1.0, .beta = 0,
    .flags = 0
};

libysmm_cl_smm_kernel_t smmk;
libysmm_cl_create_smm_kernel(&smmk, h, &smm, sizeof(smm), 0);

libysmm_cl_enqueue_smm_kernel(smmk, bufA, bufB, bufC, queue, 0, NULL, NULL);

// ...

libysmm_cl_destory_smm_kernel(smmk);
libysmm_cl_destroy_handle(h);
```
