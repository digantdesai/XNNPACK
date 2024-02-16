make -j ./packing-test && ./packing-test --gtest_filter="PACK_QD8_F32_QC4W_GEMM_GOI_W.bl_*"
make -j ./qd8-f32-qc4w-gemm-minmax-test && ./qd8-f32-qc4w-gemm-minmax-test --gtest_filter="QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR_BL*"
make -j fully-connected-nc-test && ./fully-connected-nc-test --gtest_filter="*FULLY_CONNECTED_NC_QD8_F32_QB4W*"
make -j fully-connected-test && ./fully-connected-test --gtest_filter="*FullyConnectedTestQD8F32QB4W*"
