cd ..

export HSA_DISABLE_CACHE=0
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 6 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 8 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 9 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 10 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 12 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 16 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 20 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 32 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x4 --lds 64 --clean
