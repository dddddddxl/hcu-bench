cd ..

export HSA_DISABLE_CACHE=0
./run bandwidth memory_readx4_x4 --lds 6 --clean
./run bandwidth memory_readx4_x4 --lds 8 --clean
./run bandwidth memory_readx4_x4 --lds 9 --clean
./run bandwidth memory_readx4_x4 --lds 10 --clean
./run bandwidth memory_readx4_x4 --lds 12 --clean
./run bandwidth memory_readx4_x4 --lds 16 --clean
./run bandwidth memory_readx4_x4 --lds 20 --clean
./run bandwidth memory_readx4_x4 --lds 32 --clean
./run bandwidth memory_readx4_x4 --lds 64 --clean
