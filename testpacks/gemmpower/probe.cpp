#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    hipDeviceProp_t properties{};
    char pci[64]{};
    if (hipGetDeviceCount(&count) != hipSuccess || count != 1 ||
        hipGetDeviceProperties(&properties, 0) != hipSuccess ||
        hipDeviceGetPCIBusId(pci, sizeof(pci), 0) != hipSuccess) {
        std::cerr << "Expected exactly one accessible HIP device\n";
        return 2;
    }
    std::cout << "{\"architecture\":\"" << properties.gcnArchName
              << "\",\"pci\":\"" << pci << "\",\"visible_count\":1}\n";
    return 0;
}
