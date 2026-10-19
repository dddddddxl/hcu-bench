cd ..

export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx1_x1 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx2_x1 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 --clean

numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx1_x4 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx2_x4 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --clean

export HSA_DISABLE_CACHE=1
numactl --cpubind=0 --membind=0 ./run bandwidth memory_writex1_x1 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_writex2_x1 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_write --clean

numactl --cpubind=0 --membind=0 ./run bandwidth memory_writex1_x4 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_writex2_x4 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_writex4_x4 --clean

export HSA_DISABLE_CACHE=0

