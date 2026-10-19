cd ..

numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 -w 8 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 -w 16 --clean

numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx2_x2 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx2_x2 -w 8 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx2_x2 -w 16 --clean

numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx1_x4 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx1_x4 -w 8 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx1_x4 -w 16 --clean

numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx4_x1 -w 16 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx2_x1 -w 16 --clean
numactl --cpubind=0 --membind=0 ./run bandwidth memory_readx1_x1 -w 16 --clean
