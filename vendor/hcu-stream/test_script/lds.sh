echo "memory_read"
./run bandwidth memory_readx4_x4 --lds 6 --clean
./run bandwidth memory_readx4_x4 --lds 8 --clean
./run bandwidth memory_readx4_x4 --lds 9 --clean
./run bandwidth memory_readx4_x4 --lds 10 --clean
./run bandwidth memory_readx4_x4 --lds 12 --clean
./run bandwidth memory_readx4_x4 --lds 16 --clean
./run bandwidth memory_readx4_x4 --lds 20 --clean
./run bandwidth memory_readx4_x4 --lds 32 --clean
./run bandwidth memory_readx4_x4 --lds 64 --clean
export HSA_DISABLE_CACHE=1
echo "memory_write"
./run bandwidth memory_writex4_x4 --lds 6 --clean
./run bandwidth memory_writex4_x4 --lds 8 --clean
./run bandwidth memory_writex4_x4 --lds 9 --clean
./run bandwidth memory_writex4_x4 --lds 10 --clean
./run bandwidth memory_writex4_x4 --lds 12 --clean
./run bandwidth memory_writex4_x4 --lds 16 --clean
./run bandwidth memory_writex4_x4 --lds 20 --clean
./run bandwidth memory_writex4_x4 --lds 32 --clean
./run bandwidth memory_writex4_x4 --lds 64 --clean

export HSA_DISABLE_CACHE=0
echo "memory_copy"
./run bandwidth memory_copy --lds 6 --clean
./run bandwidth memory_copy --lds 8 --clean
./run bandwidth memory_copy --lds 9 --clean
./run bandwidth memory_copy --lds 10 --clean
./run bandwidth memory_copy --lds 12 --clean
./run bandwidth memory_copy --lds 16 --clean
./run bandwidth memory_copy --lds 20 --clean
./run bandwidth memory_copy --lds 32 --clean
./run bandwidth memory_copy --lds 64 --clean

echo "matrix_readx4"
./run bandwidth matrix_read_x4 --lds 6 --clean
./run bandwidth matrix_read_x4 --lds 8 --clean
./run bandwidth matrix_read_x4 --lds 9 --clean
./run bandwidth matrix_read_x4 --lds 10 --clean
./run bandwidth matrix_read_x4 --lds 12 --clean
./run bandwidth matrix_read_x4 --lds 16 --clean
./run bandwidth matrix_read_x4 --lds 20 --clean
./run bandwidth matrix_read_x4 --lds 32 --clean
./run bandwidth matrix_read_x4 --lds 64 --clean

echo "matrix_write_x4"
./run bandwidth matrix_write_x4 --lds 6 --clean
./run bandwidth matrix_write_x4 --lds 8 --clean
./run bandwidth matrix_write_x4 --lds 9 --clean
./run bandwidth matrix_write_x4 --lds 10 --clean
./run bandwidth matrix_write_x4 --lds 12 --clean
./run bandwidth matrix_write_x4 --lds 16 --clean
./run bandwidth matrix_write_x4 --lds 20 --clean
./run bandwidth matrix_write_x4 --lds 32 --clean
./run bandwidth matrix_write_x4 --lds 64 --clean

echo "matrix_copy_x4"
./run bandwidth matrix_copy_x4 --lds 6 --clean
./run bandwidth matrix_copy_x4 --lds 8 --clean
./run bandwidth matrix_copy_x4 --lds 9 --clean
./run bandwidth matrix_copy_x4 --lds 10 --clean
./run bandwidth matrix_copy_x4 --lds 12 --clean
./run bandwidth matrix_copy_x4 --lds 16 --clean
./run bandwidth matrix_copy_x4 --lds 20 --clean
./run bandwidth matrix_copy_x4 --lds 32 --clean
./run bandwidth matrix_copy_x4 --lds 64 --clean
export HSA_DISABLE_CACHE=1
echo "lds_to_memory_x4"
./run bandwidth lds_to_memory_x4 --lds 6 --clean
./run bandwidth lds_to_memory_x4 --lds 8 --clean
./run bandwidth lds_to_memory_x4 --lds 9 --clean
./run bandwidth lds_to_memory_x4 --lds 10 --clean
./run bandwidth lds_to_memory_x4 --lds 12 --clean
./run bandwidth lds_to_memory_x4 --lds 16 --clean
./run bandwidth lds_to_memory_x4 --lds 20 --clean
./run bandwidth lds_to_memory_x4 --lds 32 --clean
./run bandwidth lds_to_memory_x4 --lds 64 --clean
export HSA_DISABLE_CACHE=0
echo "memory_to_ldsx4_x4"
./run bandwidth memory_to_ldsx4_x4 --lds 6 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 8 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 9 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 10 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 12 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 16 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 20 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 32 --clean
./run bandwidth memory_to_ldsx4_x4 --lds 64 --clean


echo "tiling_read"
./run bandwidth matrix_read --lds 6 --clean
./run bandwidth matrix_read --lds 8 --clean
./run bandwidth matrix_read --lds 9 --clean
./run bandwidth matrix_read --lds 10 --clean
./run bandwidth matrix_read --lds 12 --clean
./run bandwidth matrix_read --lds 16 --clean
./run bandwidth matrix_read --lds 20 --clean
./run bandwidth matrix_read --lds 32 --clean
./run bandwidth matrix_read --lds 64 --clean
