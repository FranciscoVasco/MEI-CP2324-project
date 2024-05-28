set(TEST_TARGET "bm_target")

set (TEST_SOURCES
        ../src/histogram_par.cpp
        ../src/histogram_eq.cpp
        ../src/histogram_cu.cu
        bm_histogram_par.cpp
        bm_histogram_par.h)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GBM_BOTH_LIBRARIES} wb Threads::Threads OpenMP::OpenMP_CXX)