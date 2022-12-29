#include <math.h>
#include <stdio.h>

#include <sys/time.h>

#include "libysmm_cl.h"

#define NREPS 20000

cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        puts("Couldn't identify a platform\n");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err < 0)
    {
        puts("Couldn't access any devices\n");
        exit(1);
    }

    return dev;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        puts("Usage: M, N, K\n");
        exit(1);
    }

    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);
    size_t K = atoi(argv[3]);

    cl_int err;
    cl_device_id dev = create_device();
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err < 0)
    {
        puts("Couldn't create a context\n");
        exit(1);
    }

    cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, &err);
    if (err < 0)
    {
        puts("Couldn't create a queue\n");
        exit(1);
    }

    float *A = calloc(M*K, sizeof(float));
    float *B = calloc(K*N, sizeof(float));
    float *C = calloc(N*M, sizeof(float));
    float *refC = calloc(N*M, sizeof(float));

    for (int i = 0; i < M*K; i++)
    {
        A[i] = (float) rand() / (float) RAND_MAX;
    }

    for (int i = 0; i < K*N; i++)
    {
        B[i] = (float) rand() / (float) RAND_MAX;
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float acc = 0;

            for (int k = 0; k < K; k++)
                acc += A[i*K + k]*B[k*N + j];

            refC[i*N + j] = acc;
        }
    }

    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K*N*sizeof(*B), NULL, &err);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M*N*sizeof(*C), NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);

    libysmm_cl_handle_t h;
    err = libysmm_cl_create_handle(&h, ctx, dev, 0);
    if (err < 0)
    {
        puts("Couldn't create a handle\n");
        exit(1);
    }

    libysmm_smm_t smm = {
        .dtype = LIBYSMM_DTYPE_FP32,
        .layout = LIBYSMM_LAYOUT_ROW_MAJOR,
        .transpose = LIBYSMM_TRANSPOSE_NN,
        .m = M, .n = N, .k = K,
        .lda = K, .ldb = N, .ldc = N,
        .alpha = 1.0, .beta = 0,
        .a = A, .flags = 0
    };

    libysmm_cl_smm_kernel_t smmk;
    err = libysmm_cl_create_smm_kernel(&smmk, h, &smm, sizeof(smm), 0);
    if (err < 0)
    {
        fprintf(stderr, "Couldn't create a kernel %d\n", err);
        exit(1);
    }

    err = libysmm_cl_bind_smm_kernel(smmk, bufB, bufC);
    if (err < 0)
    {
        perror("Couldn't bind a kernel");
        exit(1);
    }

    err = libysmm_cl_enqueue_smm_kernel(smmk, queue, 0, NULL, NULL);
    if (err < 0)
    {
        fprintf(stderr, "Couldn't enqueue a kernel %d\n", err);
        exit(1);
    }

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    if (err < 0)
    {
        perror("Couldn't read back C");
        exit(1);
    }

    double diff = -1;
    for (int i = 0; i < N*M; i++)
    {
        if (fabs(C[i] - refC[i]) > diff)
            diff = fabs(C[i] - refC[i]);
    }

    printf("Max abs difference is: %f\n", diff);

    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    for (int i = 0; i < NREPS; i++)
    {
        err = libysmm_cl_enqueue_smm_kernel(smmk, queue, 0, NULL, NULL);
        if (err < 0)
        {
            perror("Couldn't enqueue a kernel");
            exit(1);
        }
    }
    err = clFinish(queue);
    gettimeofday(&end, NULL);

    double dt = (end.tv_sec - begin.tv_sec)
              + ((end.tv_usec - begin.tv_usec) / 1000000.0);
    double gflops = NREPS*2*M*N*K / dt / 1e9;
    double gbytes = NREPS*4*(M + K)*N / dt / pow(1024, 3);

    printf("%f GFLOP/s\n%f GiB/s\n", gflops, gbytes);

    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    free(A);
    free(B);
    free(C);
    free(refC);

    libysmm_cl_destory_smm_kernel(smmk);
    libysmm_cl_destroy_handle(h);

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
