cmake_minimum_required(VERSION 3.16)
project(project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

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
        src/histogram_par.cpp
        src/histogram_eq.cpp
        include/histogram_eq.h
)

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
set (GBM_BOTH_LIBRARIES benchmark_main benchmark libbenchmark libbenchmark_main)

add_subdirectory (test)
add_subdirectory(benchmarking)



