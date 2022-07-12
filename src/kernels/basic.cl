R"(
__kernel void
mm(__global const float* restrict a,
   __global const float* restrict b,
   __global float* restrict c)
{
    const int M = {{ M }}, N = {{ N }}, K = {{ K }};
    const int lda = {{ lda }}, ldb = {{ ldb }}, ldc = {{ ldc }};
    const float alpha = {{ alpha }}, beta = {{ beta }};

    const int cx = get_global_id(0);

    // Load A into shared memory
    __local float l_a[{{ M }}][{{ K }}];

    for (int i = 0; i < M; i++)
        for (int k = get_local_id(0); k < K; k += get_local_size(0))
            l_a[i][k] = a[i*lda + k];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (cx < N)
    {
        for (int i = 0; i < M; i++)
        {
            float acc = 0;

            for (int k = 0; k < K; k++)
                acc += l_a[i][k]*b[k*ldb + cx];

            c[i*ldc + cx] = alpha*acc + beta*c[i*ldc + cx];
        }
    }
}
)"
