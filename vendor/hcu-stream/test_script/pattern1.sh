
cd ../
./run bandwidth l1_read --clean
./run bandwidth l2_read --clean
./run bandwidth l2_write --clean
./run bandwidth lds_read --clean
./run bandwidth lds_write --clean
./run bandwidth memory_readx4_x1 --clean

export HSA_DISABLE_CACHE=1
./run bandwidth memory_write --clean

export HSA_DISABLE_CACHE=0
./run bandwidth memory_copy --clean
./run bandwidth memory_to_ldsx1 --clean

export HSA_DISABLE_CACHE=1
./run bandwidth lds_to_memory --clean

export HSA_DISABLE_CACHE=0
./run bandwidth matrix_read --clean
./run bandwidth matrix_write --clean
./run bandwidth matrix_copy --clean
./run bandwidth mapping_matrix_write --clean
./run bandwidth tiling_read --clean
./run bandwidth memory_readx4_x1 -w 16 --clean

export HSA_DISABLE_CACHE=1
./run bandwidth memory_write -w 16 --clean

export HSA_DISABLE_CACHE=0
./run bandwidth memory_copy -w 16 --clean
./run bandwidth memory_readx4_x64 --clean