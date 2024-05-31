cmake_minimum_required(VERSION 3.16)
project(project LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set( CMAKE_CUDA_FLAGS "" CACHE STRING "" )

set(CMAKE_BUILD_TYPE Release)

if ( CMAKE_CUDA_FLAGS )
    list(REMOVE_ITEM CMAKE_CUDA_FLAGS "--cudart static")
endif()
string(APPEND CMAKE_CUDA_FLAGS "--cudart shared")

set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS}")

##############################################################
# Autodetect CUDA architecture(s)
##############################################################

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
message("-- CUDA Architecture(s): " ${CMAKE_CUDA_ARCHITECTURES})

###############
# libWB
###############

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
add_subdirectory(libwb)
include_directories(libwb)



###############
# project
###############

include_directories(include dataset ${SOURCE_DIR}/include)


find_package(OpenMP)
find_package(Threads REQUIRED)

add_executable(project
        src/main.cpp
        src/histogram_eq.cpp

        src/histogram_par.cpp
        src/histogram_par_f.cpp

        src/histogram_cu.cu
        src/histogram_cu_f.cu

        include/histogram_eq.h
        include/histogram_par.h
        include/histogram_par_f.h
        include/histogram_cu.cuh
        include/histogram_cu_f.cuh

)

add_executable(dataset_generator
        src/dataset_generator.cpp
        src/dataset_generator.cpp
)
target_link_libraries(dataset_generator wb)

if(OpenMP_CXX_FOUND)
    message("------ found")
    target_link_libraries(project PUBLIC wb Threads::Threads OpenMP::OpenMP_CXX)
    target_compile_options(project PUBLIC "${OpenMP_CXX_FLAGS}")
endif()

#target_link_libraries(project wb)

##############################################################
# Tests
##############################################################

enable_testing()

#############
# Download and unpack googletest at configure time
#############

set (GTEST_HOME ${CMAKE_CURRENT_BINARY_DIR}/googletest)


message ("-- Loading Google Test from ${GTEST_HOME}")

set (GBM_HOME ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark)


configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeLists.googletest.in ${GTEST_HOME}/download/CMakeLists.txt)

configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeLists.googlebenchmark.in ${GBM_HOME}/download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GBM_HOME}/download )
if(result)
    message(FATAL_ERROR "CMake step for googlebenchmark failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GBM_HOME}/download )
if(result)
    message(FATAL_ERROR "Build step for googlebenchmark failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(gbenchmark_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines the gtest and gtest_main targets.
add_subdirectory(${GTEST_HOME}/src
        ${GTEST_HOME}/build
        EXCLUDE_FROM_ALL)

add_subdirectory(${GBM_HOME}/src
        ${GBM_HOME}/build
        EXCLUDE_FROM_ALL)

set (GTEST_BOTH_LIBRARIES gtest_main gtest)
set (GBM_BOTH_LIBRARIES benchmark_main benchmark)

add_subdirectory(test)
add_subdirectory(benchmarking)



