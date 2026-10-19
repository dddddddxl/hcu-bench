cd ..

numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 --clean
export HSA_DISABLE_CACHE=1
numactl --cpubind=0 --membind=0 ./run bandwidth memory_write --clean
export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth memory_copy --clean

numactl --cpubind=0 --membind=0 ./run bandwidth matrix_read --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_write --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_copy --clean


numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 -w 8 --clean
export HSA_DISABLE_CACHE=1
numactl --cpubind=0 --membind=0 ./run bandwidth memory_write -w 8 --clean
export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth memory_copy -w 8 --clean

numactl --cpubind=0 --membind=0 ./run bandwidth matrix_read -w 8 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_write -w 8 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_copy -w 8 --clean

numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 -w 16 --clean
export HSA_DISABLE_CACHE=1
numactl --cpubind=0 --membind=0 ./run bandwidth memory_write -w 16 --clean
export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth memory_copy -w 16 --clean


numactl --cpubind=0 --membind=0 ./run bandwidth matrix_read -w 16 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_write -w 16 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth matrix_copy -w 16 --clean
