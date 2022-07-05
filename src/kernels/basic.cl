R"(
__kernel void
mm(__global const float* restrict a,
   __global const float* restrict b,
   __global float* restrict c)
{
    const int M = {{ M }}, N = {{ N }}, K = {{ K }};
    const int lda = {{ lda }}, ldb = {{ ldb }}, ldc = {{ ldc }};
    const int rx = get_global_id(0);
    const int cx = get_global_id(1);
    const float alpha = {{ alpha }}, beta = {{ beta }};

    if (rx < M && cx < N)
    {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += a[rx*lda + k] * b[k*ldb + cx];

        c[rx*ldc + cx] = alpha*acc + beta*c[rx*ldc + cx];
    }
}
)"
