cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(run_project CXX)

# ============================
# ROCM PATH
# ============================
if (NOT ROCM_PATH)
    set(ROCM_PATH /opt/rocm)
endif()
message(STATUS "ROCM_PATH = ${ROCM_PATH}")

set(ROCM_INCLUDE_DIRS ${ROCM_PATH}/include)
set(ROCM_LIBRARY_DIRS ${ROCM_PATH}/lib)

# ============================
# Use hipcc compiler
# ============================
set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
add_definitions(-D__HIP_PLATFORM_AMD__)

include_directories(
    ${ROCM_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/kernel
    ${ROCM_PATH}/include/hiprand
)

link_directories(${ROCM_LIBRARY_DIRS})

# ============================
# option : multi compile args
# ============================
option(MULTI_ARCH_BUILD "Build for multiple GPU architectures" OFF)

# support GPU_ARCH
set(SUPPORTED_ARCHS 
    "gfx926:926" 
    "gfx928:928"
    "gfx936:936"
    "gfx938:938"
)

# ============================
# single compile
# ============================
if (NOT MULTI_ARCH_BUILD)
    if (NOT GPU_ARCH)
        message(FATAL_ERROR "GPU_ARCH not specified. Use -DGPU_ARCH=gfx926 or gfx938 etc.")
    endif()
    
    if (NOT GPU_ARCH_NUM)
        message(FATAL_ERROR "GPU_ARCH_NUM not specified. Example: -DGPU_ARCH_NUM=926")
    endif()
    
    message(STATUS "GPU_ARCH = ${GPU_ARCH}")
    message(STATUS "GPU_ARCH_NUM = ${GPU_ARCH_NUM}")
    
    # Definitions equivalent to Makefile
    add_definitions(-DHIP_ARCH=${GPU_ARCH_NUM})
    set(OFFLOAD_FLAG "--offload-arch=${GPU_ARCH}")
    

    set(EXECUTABLE_NAME "hcu-stream")
    
    # ============================
    # compiler option
    # ============================
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -Wall")
    
    # DIRPATH macro
    add_definitions(-DDIRPATH=\"${PROJECT_SOURCE_DIR}\")
    
    # ============================
    # 源文件
    # ============================
    set(SRC_FILES 
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/BWtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/output.cpp
        ${PROJECT_SOURCE_DIR}/src/device.cpp
        ${PROJECT_SOURCE_DIR}/src/latency/LTtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/latency/Dispatchtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/compute/TFtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/L2cacheRW.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/TCCRead.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/element_wise.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/gemmMemory.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/faMemory.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/paMemory.cpp
        ${PROJECT_SOURCE_DIR}/src/UnitTest/VgprFlush.cpp
    )
    
    # ============================
    # binary name
    # ============================
    add_executable(${EXECUTABLE_NAME} ${SRC_FILES})
    
    target_compile_options(${EXECUTABLE_NAME} PRIVATE ${OFFLOAD_FLAG})
    target_link_libraries(${EXECUTABLE_NAME} amdhip64 m stdc++)
    
    # path
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    
# ============================
# 多架构编译模式
# ============================
else()
    message(STATUS "Building for multiple GPU architectures")
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -Wall -w")
    add_definitions(-DDIRPATH=\"${PROJECT_SOURCE_DIR}\")
    
    # ============================
    # 源文件
    # ============================
    set(SRC_FILES 
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/BWtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/output.cpp
        ${PROJECT_SOURCE_DIR}/src/device.cpp
        ${PROJECT_SOURCE_DIR}/src/latency/LTtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/latency/Dispatchtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/compute/TFtestcase.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/L2cacheRW.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/TCCRead.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/element_wise.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/gemmMemory.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/faMemory.cpp
        ${PROJECT_SOURCE_DIR}/src/memory/paMemory.cpp
        ${PROJECT_SOURCE_DIR}/src/UnitTest/VgprFlush.cpp
    )
    
    # 为每个架构创建独立的可执行文件
    foreach(ARCH_PAIR ${SUPPORTED_ARCHS})
        string(REPLACE ":" ";" ARCH_LIST ${ARCH_PAIR})
        list(GET ARCH_LIST 0 GPU_ARCH)
        list(GET ARCH_LIST 1 GPU_ARCH_NUM)
        
        message(STATUS "Adding target for ${GPU_ARCH} (arch num: ${GPU_ARCH_NUM})")
        
        # 创建架构特定的可执行文件名
        set(TARGET_NAME "hcu-stream-${GPU_ARCH}")
        
        add_executable(${TARGET_NAME} ${SRC_FILES})
        
        target_compile_definitions(${TARGET_NAME} PRIVATE -DHIP_ARCH=${GPU_ARCH_NUM})
        target_compile_options(${TARGET_NAME} PRIVATE "--offload-arch=${GPU_ARCH}")
        
        target_link_libraries(${TARGET_NAME} amdhip64 m stdc++)
        
        set_target_properties(${TARGET_NAME} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endforeach()
    
    # ============================
    # ============================
    option(BUILD_FATBINARY "Build a fatbinary containing all architectures" OFF)
    
    if(BUILD_FATBINARY)
        message(STATUS "Building fatbinary for all architectures")
        
        set(FATBINARY_NAME "hcu-stream")
        
        # 收集所有架构的 offload-arch 标志
        set(ALL_OFFLOAD_FLAGS)
        foreach(ARCH_PAIR ${SUPPORTED_ARCHS})
            string(REPLACE ":" ";" ARCH_LIST ${ARCH_PAIR})
            list(GET ARCH_LIST 0 GPU_ARCH)
            list(APPEND ALL_OFFLOAD_FLAGS "--offload-arch=${GPU_ARCH}")
        endforeach()
        
        # 创建通用可执行文件
        add_executable(${FATBINARY_NAME} ${SRC_FILES})
        
        # 添加所有架构的编译标志
        target_compile_options(${FATBINARY_NAME} PRIVATE ${ALL_OFFLOAD_FLAGS})
        
        # 使用第一个架构的 arch num 作为定义
        list(GET SUPPORTED_ARCHS 0 FIRST_ARCH)
        string(REPLACE ":" ";" FIRST_ARCH_LIST ${FIRST_ARCH})
        list(GET FIRST_ARCH_LIST 1 FIRST_ARCH_NUM)
        target_compile_definitions(${FATBINARY_NAME} PRIVATE -DHIP_ARCH=${FIRST_ARCH_NUM})
        
        target_link_libraries(${FATBINARY_NAME} amdhip64 m stdc++)
        
        set_target_properties(${FATBINARY_NAME} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endif()
endif()

# ============================
# 安装目标（可选）
# ============================
install(TARGETS ${EXECUTABLE_NAME} 
        RUNTIME DESTINATION bin
        CONFIGURATIONS Release)
