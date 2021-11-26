#include <xnnpack/assembly.h>

# void transpose_32bit_aarch64(
#    void *input,  x0
#    void *output, x1
#    int h,        x2
#    int w)        x3
BEGIN_FUNCTION transpose_32bit_aarch64
    # Shift height and width to get the number of bytes per row/column.
    lsl  x2, x2, 2
    lsl  x3, x3, 2

    # Set up the pointers for loading each row.
    add  x4, x0, x3
    add  x5, x4, x3
    add  x6, x5, x3

    # Set up the pointers for storing.
    add  x7, x2, x2
    add  x8, x7, x2

    ldp  q0, q1, [x0]
    ldp  q2, q3, [x4]
    ldp  q4, q5, [x5]
    ldp  q6, q7, [x6]

    zip1 v8.4s, v0.4s, v4.4s
    zip2 v9.4s, v0.4s, v4.4s
    zip1 v10.4s, v2.4s, v6.4s
    zip2 v11.4s, v2.4s, v6.4s

    zip1 v0.4s, v8.4s, v10.4s
    zip2 v2.4s, v8.4s, v10.4s
    zip1 v4.4s, v9.4s, v11.4s
    zip2 v6.4s, v9.4s, v11.4s

    zip1 v8.4s, v1.4s, v5.4s
    zip2 v9.4s, v1.4s, v5.4s
    zip1 v10.4s, v3.4s, v7.4s
    zip2 v11.4s, v3.4s, v7.4s

    zip1 v1.4s, v8.4s, v10.4s
    zip2 v3.4s, v8.4s, v10.4s
    zip1 v5.4s, v9.4s, v11.4s
    zip2 v7.4s, v9.4s, v11.4s

    str q0, [x1]
    str q2, [x1, x2]
    str q4, [x1, x7]
    str q6, [x1, x8]

    # Add 4 rows to the output pointer.
    mov x10, 4
    mul x11, x10, x2
    add x11, x11, x1

    str q1, [x11]
    str q3, [x11, x2]
    str q5, [x11, x7]
    str q7, [x11, x8]

    # Increment input pointer by 4 rows.
    lsl x12, x3, 2
    add x0, x0, x12

    # Set up the pointers for loading each row.
    add  x4, x0, x3
    add  x5, x4, x3
    add  x6, x5, x3

    ldp  q0, q1, [x0]
    ldp  q2, q3, [x4]
    ldp  q4, q5, [x5]
    ldp  q6, q7, [x6]

    zip1 v8.4s, v0.4s, v4.4s
    zip2 v9.4s, v0.4s, v4.4s
    zip1 v10.4s, v2.4s, v6.4s
    zip2 v11.4s, v2.4s, v6.4s

    zip1 v0.4s, v8.4s, v10.4s
    zip2 v2.4s, v8.4s, v10.4s
    zip1 v4.4s, v9.4s, v11.4s
    zip2 v6.4s, v9.4s, v11.4s

    zip1 v8.4s, v1.4s, v5.4s
    zip2 v9.4s, v1.4s, v5.4s
    zip1 v10.4s, v3.4s, v7.4s
    zip2 v11.4s, v3.4s, v7.4s

    zip1 v1.4s, v8.4s, v10.4s
    zip2 v3.4s, v8.4s, v10.4s
    zip1 v5.4s, v9.4s, v11.4s
    zip2 v7.4s, v9.4s, v11.4s

    # The output pointers are incremented by 16 bytes.
    mov x10, 16
    add x1, x1, 16
    add x11, x11, 16
    str q0, [x1]
    str q2, [x1, x2]
    str q4, [x1, x7]
    str q6, [x1, x8]

    str q1, [x11]
    str q3, [x11, x2]
    str q5, [x11, x7]
    str q7, [x11, x8]

    RET
END_FUNCTION transpose_32bit_aarch64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
