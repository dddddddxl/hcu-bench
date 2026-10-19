
cd ../
numactl --cpubind=0 --membind=0 ./run bandwidth l1_read --clean
numactl --cpubind=0 --membind=0 ./run bandwidth l2_read --clean
numactl --cpubind=0 --membind=0 ./run bandwidth l2_write --clean
numactl --cpubind=0 --membind=0 ./run bandwidth lds_read --clean
numactl --cpubind=0 --membind=0 ./run bandwidth lds_write --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 --clean

export HSA_DISABLE_CACHE=1
numactl --cpubind=0 --membind=0 ./run bandwidth memory_write --clean

export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth memory_copy --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_to_ldsx1 --clean

export HSA_DISABLE_CACHE=1
numactl --cpubind=0 --membind=0 ./run bandwidth lds_to_memory --clean

export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_read --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_write --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_copy --clean
numactl --cpubind=0 --membind=0 ./run bandwidth mapping_matrix_write --clean
numactl --cpubind=0 --membind=0 ./run bandwidth tiling_read --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 -w 16 --clean

export HSA_DISABLE_CACHE=1
numactl --cpubind=0 --membind=0 ./run bandwidth memory_write -w 16 --clean

export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth memory_copy -w 16 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x64 --clean

