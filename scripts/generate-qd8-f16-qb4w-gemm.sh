# TODO Merge this logic into the main gemm generator script.

tools/xngen src/qs8-gemm/scalar-qd8-f16-qb4w.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x2-minmax-scalar.c
tools/xngen src/qs8-gemm/scalar-qd8-f16-qb4w.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x4-minmax-scalar.c
tools/xngen src/qs8-gemm/scalar-qd8-f16-qb4w.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8-minmax-scalar.c