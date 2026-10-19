cd ..

export HSA_DISABLE_CACHE=0
./run bandwidth memory_readx4_x1 --clean
./run bandwidth memory_readx4_x1 -w 8 --clean
./run bandwidth memory_readx4_x1 -w 16 --clean
./run bandwidth matrix_read --clean
./run bandwidth matrix_read -w 8 --clean
./run bandwidth matrix_read -w 16 --clean

export HSA_DISABLE_CACHE=1
./run bandwidth memory_write --clean
./run bandwidth memory_write -w 8 --clean
./run bandwidth memory_write -w 16 --clean

export HSA_DISABLE_CACHE=0
./run bandwidth matrix_write --clean
./run bandwidth matrix_write -w 8 --clean
./run bandwidth matrix_write -w 16 --clean


./run bandwidth memory_copy --clean
./run bandwidth memory_copy -w 8 --clean
./run bandwidth memory_copy -w 16 --clean
./run bandwidth matrix_copy --clean
./run bandwidth matrix_copy -w 8 --clean
./run bandwidth matrix_copy -w 16 --clean
