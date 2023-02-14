R"(
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define block_read4f(p) as_float4(intel_sub_group_block_read4((__global uint *) p))
#define block_write4f(p, v) intel_sub_group_block_write4((__global uint *) p, as_uint4(v))

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void
mm(__global const float* restrict a,
   __global const float* restrict b,
   __global float* restrict c,
   int m, int n, int k,
   int lda, int ldb, int ldc)
{
    // Each thread does sixteen rows
    int g_row = 16*get_global_id(1);

    // Each sub-group of eight threads does 32 columns
    int g_col = 32*get_global_id(0) / 8;

    // If we have no work to do then return
    if (g_col >= n || g_row >= m)
        return;

    // Displacements
    uint aoff = g_row*lda;
    uint boff = g_col;
    uint coff = g_row*ldc + g_col;

    float4 a_sub[16], b_sub[4], c_acc[16], temp;

    for (int i = 0; i < 16; i++)
        c_acc[i] = 0;

## if m_mod_16
    // Full M tile with 16 rows
    if (g_row + 16 < m)
## endif
    {
        for (int tk = 0; tk < k / 4; tk++, aoff += 32)
        {
            temp = block_read4f(a + aoff);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                a_sub[i] = intel_sub_group_shuffle(temp, i);

            temp = block_read4f(a + aoff + 8*lda);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                a_sub[i + 8] = intel_sub_group_shuffle(temp, i);

            #pragma unroll
            for (int i = 0; i < 4; i++, boff += ldb)
                b_sub[i] = block_read4f(b + boff);

## for p in range(4)
            #pragma unroll
            for (int i = 0; i < 16; i++)
                c_acc[i] += a_sub[i].s{{p}}*b_sub[{{p}}];
## endfor
        }

## if k_mod_4
        // Partial K tile with {{k_mod_4}} columns
        temp = block_read4f(a + aoff);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            a_sub[i] = intel_sub_group_shuffle(temp, i);

        temp = block_read4f(a + aoff + 8*lda);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            a_sub[i + 8] = intel_sub_group_shuffle(temp, i);

        #pragma unroll
        for (int i = 0; i < {{k_mod_4}}; i++, boff += ldb)
            b_sub[i] = block_read4f(b + boff);

## for p in range(k_mod_4)
        #pragma unroll
        for (int i = 0; i < 16; i++)
            c_acc[i] += a_sub[i].s{{p}}*b_sub[{{p}}];
## endfor
## endif

        // Write out the result
        #pragma unroll
        for (int i = 0; i < 16; i++, coff += ldc)
## if beta == 0
            block_write4f(c + coff, c_acc[i]);
## else if beta == 1
            block_write4f(c + coff, c_acc[i] + block_read4f(c + coff));
## else
            block_write4f(c + coff, c_acc[i] + {{beta}}*block_read4f(c + coff));
## endif

    }
## if m_mod_16
    // Partial M tile with {{m_mod_16}} rows
    else
    {
        for (int tk = 0; tk < k / 4; tk++, aoff += 32)
        {
            temp = block_read4f(a + aoff);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                a_sub[i] = intel_sub_group_shuffle(temp, i);

## if m_mod_16 > 8
            temp = block_read4f(a + aoff + 8*lda);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                a_sub[i + 8] = intel_sub_group_shuffle(temp, i);
## endif

            #pragma unroll
            for (int i = 0; i < 4; i++, boff += ldb)
                b_sub[i] = block_read4f(b + boff);

## for p in range(4)
            #pragma unroll
            for (int i = 0; i < {{m_mod_16}}; i++)
                c_acc[i] += a_sub[i].s{{p}}*b_sub[{{p}}];
## endfor
        }

## if k_mod_4
        // Partial K tile with {{k_mod_4}} columns
        temp = block_read4f(a + aoff);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            a_sub[i] = intel_sub_group_shuffle(temp, i);
## if m_mod_16 > 8
        temp = block_read4f(a + aoff + 8*lda);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            a_sub[i + 8] = intel_sub_group_shuffle(temp, i);
## endif

        #pragma unroll
        for (int i = 0; i < {{k_mod_4}}; i++, boff += ldb)
            b_sub[i] = block_read4f(b + boff);

## for p in range(k_mod_4)
        #pragma unroll
        for (int i = 0; i < {{m_mod_16}}; i++)
            c_acc[i] += a_sub[i].s{{p}}*b_sub[{{p}}];

## endfor
## endif

        // Write out the result
        #pragma unroll
        for (int i = 0; i < {{m_mod_16}}; i++, coff += ldc)
## if beta == 0
            block_write4f(c + coff, c_acc[i]);
## else if beta == 1
            block_write4f(c + coff, c_acc[i] + block_read4f(c + coff));
## else
            block_write4f(c + coff, c_acc[i] + {{beta}}*block_read4f(c + coff));
## endif
    }
## endif
}
)"
