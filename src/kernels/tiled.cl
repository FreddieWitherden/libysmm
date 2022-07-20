R"(
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define block_read4f(p) as_float4(intel_sub_group_block_read4((__global uint *) p))
#define block_write4f(p, v) intel_sub_group_block_write4((__global uint *) p, as_uint4(v))

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void
mm(__global const float* restrict a,
   __global const float* restrict b,
   __global float* restrict c, int k)
{
    const int lda = {{ lda }}, ldb = {{ ldb }}, ldc = {{ ldc }};

    // Each thread does sixteen rows
    int g_row = 16*get_global_id(1);

    // Each sub-group of eight threads does 32 columns
    int g_col = 32*get_global_id(0) / 8;

    // Pre-displace our arguments
    a += g_row*lda;
    b += g_col;
    c += g_row*ldc + g_col;

    float4 a_sub[16], b_sub[4], c_acc[16], temp;

    for (int i = 0; i < 16; i++)
        c_acc[i] = 0;

    // Loop over the tiles of A in (M, dK) = (16, 4)
    for (int tk = 0; tk < k / 4; tk++, a += 32)
    {
        temp = block_read4f(a);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            a_sub[i] = intel_sub_group_shuffle(temp, i);

        temp = block_read4f(a + 8*lda);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            a_sub[i + 8] = intel_sub_group_shuffle(temp, i);

        #pragma unroll
        for (int i = 0; i < 4; i++, b += ldb)
            b_sub[i] = block_read4f(b);

        #pragma unroll
        for (int i = 0; i < 16; i++)
            c_acc[i] += a_sub[i].s0*b_sub[0];

        #pragma unroll
        for (int i = 0; i < 16; i++)
            c_acc[i] += a_sub[i].s1*b_sub[1];

        #pragma unroll
        for (int i = 0; i < 16; i++)
            c_acc[i] += a_sub[i].s2*b_sub[2];

        #pragma unroll
        for (int i = 0; i < 16; i++)
            c_acc[i] += a_sub[i].s3*b_sub[3];
    }

    // Write out the result
    #pragma unroll
    for (int i = 0; i < 16; i++, c += ldc)
        block_write4f(c, c_acc[i]);
}
)"
