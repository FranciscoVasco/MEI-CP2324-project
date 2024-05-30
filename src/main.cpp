#include "histogram_eq.h"
#include "histogram_cu.cuh"
#include "histogram_par.h"
#include <cstdlib>

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
    std::cout << n_iterations << std::endl;
    wbImage_t outputImage =cuda::iterative_histogram_equalization(inputImage,n_iterations);
    std::cout << argv[3] << std::endl;
    wbExport(argv[3], outputImage);
    //cuda::test2();
    return 0;
}