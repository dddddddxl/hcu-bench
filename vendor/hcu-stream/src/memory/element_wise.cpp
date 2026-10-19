#include "element_wise.hpp"
#include "testcase.hpp"

ElementWise::ElementWise(Output *output, Device *device, bool clean)
    : MemoryBandwidth({"elewise"}, /*wave*/ 4, /*crop*/ 1, output, device,
                      /*lds_size*/ 0, clean, 0, false, false) {
  data_shape = 256;
}

  void ElementWise::runAll() {
    test_elementwise_modern();
    test_elementwise_legacy();
  }
  
  void ElementWise::run() {
    run_benchmark<float>(elementwise_add_f32x4_kernel, "f32x4", 4, false);
    run_benchmark<half>(elementwise_add_f16x4_kernel, "f16x4", 4, false);
    printf("half vec_size = 4\n");
    run_benchmark_legacy<half, 256, 4>(4, false);
    printf("float vec_size = 4\n");
    run_benchmark_legacy<float, 256, 4>(4, false);
  }

  void ElementWise::test_elementwise_modern() {
    printf("----------------------------------------------------------\n");
    printf("----------------test for modern elementwise---------------\n");
    printf("----------------------------------------------------------\n");
    run_benchmark<float>(elementwise_add_f32_kernel, "f32", 1);
    run_benchmark<float>(elementwise_add_f32x2_kernel, "f32x2", 2);
    run_benchmark<float>(elementwise_add_f32x4_kernel, "f32x4", 4);
    run_benchmark<half>(elementwise_add_f16_kernel, "f16", 1);
    run_benchmark<half>(elementwise_add_f16x2_kernel, "f16x2", 2);
    run_benchmark<half>(elementwise_add_f16x4_kernel, "f16x4", 4);
  }

  void ElementWise::test_elementwise_legacy() {
    printf("----------------------------------------------------------\n");
    printf("----------------test for legacy elementwise---------------\n");
    printf("----------------------------------------------------------\n");
    printf("half vec_size = 1\n");
    run_benchmark_legacy<half, 256, 4>(1);
    printf("half vec_size = 2\n");
    run_benchmark_legacy<half, 256, 4>(2);
    printf("half vec_size = 4\n");
    run_benchmark_legacy<half, 256, 4>(4);
    printf("float vec_size = 1\n");
    run_benchmark_legacy<float, 256, 4>(1);
    printf("float vec_size = 2\n");
    run_benchmark_legacy<float, 256, 4>(2);
    printf("float vec_size = 4\n");
    run_benchmark_legacy<float, 256, 4>(4);
  }

  template <typename T>
  void ElementWise::run_benchmark(void (*kernel)(T *, T *, T *, int),
                                  const char *tag, int elenum, bool run_all) {
    std::vector<int> hidden, h_temp = {256, 1024, 2048, 4096};
    std::vector<int> token_num, t_temp = {32, 128, 256, 1024, 4096, 8192};
    if (run_all) {
      hidden = h_temp;
      token_num = t_temp;
    }
    else {
      hidden.push_back(4096);
      token_num.push_back(8192);
    }
    
    T *a, *b, *c;
    printf("-------%s--------------\n", tag);
    for (int i = 0; i < hidden.size(); i++) {
      for (int j = 0; j < token_num.size(); j++) {
        int N = hidden[i] * token_num[j];

        hipMalloc(&a, N * sizeof(T));
        hipMalloc(&b, N * sizeof(T));
        hipMalloc(&c, N * sizeof(T));
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        dim3 block(256);
        dim3 grid((N + 256 - 1) / 256 / elenum);

        hipEventRecord(start);
        for (int i = 0; i < 1000; i++) {
          hipLaunchKernelGGL(kernel, grid, block, 0, 0, a, b, c, N);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipFree(a);
        hipFree(b);
        hipFree(c);
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        double bw = (N * sizeof(T) * 3. / 1024 / 1024 / 1024) /
                    (milliseconds / 1000 / 1000);
        if (use_clean) {
          printf("%.2f\n", bw);
        } else {
          printf("shape: hidden[%d] * token[%d] bandwidth: %.2f GB/s\n",
                 hidden[i], token_num[j], bw);
        }

        hipEventDestroy(start);
        hipEventDestroy(stop);
      }
    }
  }

  template <typename DT, int BLOCK_SZ, int LOOP>
  void ElementWise::run_benchmark_legacy(int vec_size, bool run_all) {
    std::vector<int> hidden, h_temp = {256, 1024, 2048, 4096};
    std::vector<int> token_num, t_temp = {32, 128, 256, 1024, 4096, 8192};
    if (run_all) {
      hidden = h_temp;
      token_num = t_temp;
    }
    else {
      hidden.push_back(4096);
      token_num.push_back(8192);
    }

    for (int i = 0; i < hidden.size(); i++) {
      for (int j = 0; j < token_num.size(); j++) {
        int pad = 1024;
        int totalElements = token_num[j] * (hidden[i] + pad);
        int numElements = token_num[j] * hidden[i];

        size_t size = totalElements * sizeof(DT);
        DT *h_input1 = (DT *)malloc(size);
        DT *h_input2 = (DT *)malloc(size);
        DT *h_output = (DT *)malloc(size);

        for (int i = 0; i < totalElements; ++i) {
          h_input1[i] = (DT)(rand() / RAND_MAX);
          h_input2[i] = (DT)(rand() / RAND_MAX);
        }

        DT *d_input1 = NULL;
        DT *d_input2 = NULL;
        DT *d_output = NULL;

        HIP_CHECK(hipMalloc((void **)&d_input1, size));
        HIP_CHECK(hipMalloc((void **)&d_input2, size));
        HIP_CHECK(hipMalloc((void **)&d_output, size));

        // printf("Copy input data from the host memory to the device\n");
        HIP_CHECK(hipMemcpy(d_input1, h_input1, size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_input2, h_input2, size, hipMemcpyHostToDevice));

        std::vector<int> in_shape = {hidden[i], token_num[j]};
        std::vector<uint64_t> in_stride = {1, (uint64_t)hidden[i] + pad};

        auto offset_calc = OffsetCalculator<uint32_t>(
            in_shape.size(), in_shape.data(), in_stride.data());

        int elementsPerBlock = BLOCK_SZ * LOOP;
        int grid = (numElements + elementsPerBlock - 1) / elementsPerBlock;
        // printf("Kernel launch with %d blocks of %d threads\n", grid,
        // BLOCK_SZ);

        const int warm = 20;
        const int ite = 5000;
        struct timeval start, end;

        switch (vec_size) {
        case 1: {
          // 1. normal elementwise
          // warm up
          for (int i = 0; i < warm; i++) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(elementwise_add<DT>), dim3(grid),
                               dim3(BLOCK_SZ), 0, 0, d_input1, d_input2,
                               d_output, numElements, LOOP, offset_calc);
          }

          HIP_CHECK(hipDeviceSynchronize());
          gettimeofday(&start, NULL);
          for (int i = 0; i < ite; i++) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(elementwise_add<DT>), dim3(grid),
                               dim3(BLOCK_SZ), 0, 0, d_input1, d_input2,
                               d_output, numElements, LOOP, offset_calc);
          }
          HIP_CHECK(hipDeviceSynchronize());
          gettimeofday(&end, NULL);

          float baseline_time = ((end.tv_sec - start.tv_sec) * 1000000. +
                                 (end.tv_usec - start.tv_usec) * 1.) /
                                ite;

          float bw = (numElements * sizeof(DT) * 3. / 1024 / 1024 / 1024) /
                     (baseline_time / 1000 / 1000);

          if (use_clean) {
            printf("%.2f\n", bw);
          } else {
            printf("shape: hidden[%d] * token[%d] bandwidth: %.2f GB/s\n",
                   hidden[i], token_num[j], bw);
          }
        } break;
        case 2: {
          // 2. vect elementwise
          // warm up
          for (int i = 0; i < warm; i++) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(elementwise_add_vect<2, DT>),
                               dim3(grid), dim3(BLOCK_SZ), 0, 0, d_input1,
                               d_input2, d_output, numElements, LOOP,
                               offset_calc);
          }

          HIP_CHECK(hipDeviceSynchronize());
          gettimeofday(&start, NULL);
          for (int i = 0; i < ite; i++) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(elementwise_add_vect<2, DT>),
                               dim3(grid), dim3(BLOCK_SZ), 0, 0, d_input1,
                               d_input2, d_output, numElements, LOOP,
                               offset_calc);
          }
          HIP_CHECK(hipDeviceSynchronize());
          gettimeofday(&end, NULL);

          float baseline_time = ((end.tv_sec - start.tv_sec) * 1000000. +
                                 (end.tv_usec - start.tv_usec) * 1.) /
                                ite;

          float bw = (numElements * sizeof(DT) * 3. / 1024 / 1024 / 1024) /
                     (baseline_time / 1000 / 1000);
          if (use_clean) {
            printf("%.2f\n", bw);
          } else {
            printf("shape: hidden[%d] * token[%d] bandwidth: %.2f GB/s\n",
                   hidden[i], token_num[j], bw);
          }
        } break;

        case 4: {
          // 3. vect 4 elementwise
          for (int i = 0; i < warm; i++) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(elementwise_add_vect<4, DT>),
                               dim3(grid), dim3(BLOCK_SZ), 0, 0, d_input1,
                               d_input2, d_output, numElements, LOOP,
                               offset_calc);
          }

          HIP_CHECK(hipDeviceSynchronize());
          gettimeofday(&start, NULL);
          for (int i = 0; i < ite; i++) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(elementwise_add_vect<4, DT>),
                               dim3(grid), dim3(BLOCK_SZ), 0, 0, d_input1,
                               d_input2, d_output, numElements, LOOP,
                               offset_calc);
          }
          HIP_CHECK(hipDeviceSynchronize());
          gettimeofday(&end, NULL);

          float baseline_time = ((end.tv_sec - start.tv_sec) * 1000000. +
                                 (end.tv_usec - start.tv_usec) * 1.) /
                                ite;
          // printf("[INFO] vect elementwise: dataType[half 4] baseline time:
          // %.2f us\n",
          //      baseline_time);

          float bw = (numElements * sizeof(DT) * 3. / 1024 / 1024 / 1024) /
                     (baseline_time / 1000 / 1000);
          if (use_clean) {
            printf("%.2f\n", bw);
          } else {
            printf("shape: hidden[%d] * token[%d] bandwidth: %.2f GB/s\n",
                   hidden[i], token_num[j], bw);
          }
          hipError_t err = hipGetLastError();
          if (err != hipSuccess) {
            fprintf(
                stderr,
                "Failed to launch elementwise_add kernel (error code %s)!\n",
                hipGetErrorString(err));
            exit(EXIT_FAILURE);
          }
        } break;

        default:
          assert(0 && "unsupport vec size!!");
        }

        // 释放内存
        HIP_CHECK(hipFree(d_input1));
        HIP_CHECK(hipFree(d_input2));
        HIP_CHECK(hipFree(d_output));
        free(h_input1);
        free(h_input2);
        free(h_output);
      }
    }
  }

void ElementWise::getList() {
  std::cout << std::left << std::setw(30) << "element-wise case" << std::endl;
  std::cout << std::left << std::setw(30)
            << "Usage: ./run elewise" << std::endl;
  std::cout << "----------------------" << std::endl;
}

void ElementWise::print() { // output->print_record();
}
