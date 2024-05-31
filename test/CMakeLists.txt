set(TEST_TARGET "template_test")

set (TEST_SOURCES
        template_test.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES})


set(TEST_TARGET "histogram_eq_test")

set (TEST_SOURCES
        histogram_eq_test.cpp
        ../src/histogram_par.cpp
        ../src/histogram_eq.cpp
        ../src/histogram_cu.cu
        ../src/histogram_cu_f.cu
        ../src/histogram_par_f.cpp
)




add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb Threads::Threads OpenMP::OpenMP_CXX)