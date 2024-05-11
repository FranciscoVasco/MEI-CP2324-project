#include <filesystem>
#include "gtest/gtest.h"
#include "histogram_par.h"

using namespace cp_par;

#define DATASET_FOLDER "../../dataset/"

TEST(HistogramEq, Input01_4) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");
    wbImage_t outputImage = iterative_histogram_equalization(inputImage, 4);
    // check if the output image is correct
}