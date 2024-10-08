//
// Created by tomedias on 5/17/24.
//
#include "wb.h"
#define LOG(message) {  std::cout << "[INFO] " <<  message << std::endl; }
#ifndef PROJECT_HISTOGRAM_CU_F_CUH
#define PROJECT_HISTOGRAM_CU_F_CUH
namespace cuda_f {

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations);
}
#endif //PROJECT_HISTOGRAM_CU_F_CUH
