R"(
#define M {{ M }}
#define N {{ N }}
#define K {{ K }}
#define LDA {{ lda }}
#define LDB {{ ldb }}
#define LDC {{ ldc }}

#define TM {{ TM }}

__kernel void
mm(__global const float* restrict a,
   __global const float* restrict b,
   __global float* restrict c)
{
    const float alpha = {{ alpha }};

    const int cx = get_global_id(0);
    const int l_idx = get_local_id(0);
    const int l_stp = get_local_size(0);

    __local float l_a[TM][K];

    // Loop over the rows of A
    for (int i = 0; i < M; i += TM)
    {
        // Load a block of up to TM rows
        for (int k = i*LDA + l_idx;
             k < min(M, i + TM)*LDA;
             k += l_stp)
        {
            float av = a[k];
            if (k % LDA < K)
                l_a[k / LDA - i][k % LDA] = av;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (cx < N)
        {
            for (int j = i; j < min(M, i + TM); j++)
            {
                float acc = 0;

                for (int k = 0; k < K; k++)
                    acc += l_a[j - i][k]*b[k*LDB + cx];

## if beta == 0
                c[j*LDC + cx] = alpha*acc;
## else if beta == 1
                c[j*LDC + cx] += alpha*acc;
## else
                c[j*LDC + cx] = alpha*acc + {{beta}}*c[j*LDC + cx];
## endif
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)"
