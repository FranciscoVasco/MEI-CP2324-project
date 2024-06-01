#include "histogram_eq.h"
#include "histogram_cu.cuh"
#include "histogram_par.h"
#include "histogram_cu_f.cuh"
#include "histogram_par_f.h"
#include <cstdlib>
#include <chrono>

int main(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "usage" << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }

    wbImage_t inputImage = wbImport(argv[1]);
//    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
//    wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations);
//    wbExport(argv[3], outputImage);


    //cuda::iterative_histogram_equalization(inputImage);
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    auto now = std::chrono::system_clock::now();
    wbImage_t outputImage =cuda_f::iterative_histogram_equalization(inputImage,n_iterations);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - now;
    std::cout << "Time taken by the operation: " << duration.count() << " seconds" << std::endl;
    return 0;
}