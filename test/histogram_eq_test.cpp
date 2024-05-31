#include <filesystem>
#include "gtest/gtest.h"
#include "histogram_par.h"
#include "histogram_eq.h"
#include "histogram_cu.cuh"
#include "histogram_par_f.h"
#include "histogram_cu_f.cuh"
#include <wb.h>

//using namespace cp_par;

#define DATASET_FOLDER "../dataset/"

TEST(HistogramEq, Input01_4) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "image2048.ppm");
    wbImage_t outputImagePar = cuda_f::iterative_histogram_equalization(inputImage,1);
    wbImage_t outputImageSeq = cp::iterative_histogram_equalization(inputImage, 1);

    const int imageWidth = wbImage_getWidth(outputImagePar);
    const int imageHeight = wbImage_getHeight(outputImagePar);

    float* imageParData = wbImage_getData(outputImagePar);
    float* imageSeqData = wbImage_getData(outputImageSeq);

    for(int i = 0; i < imageWidth; i++)
        for(int j = 0; j < imageHeight; j++)
            EXPECT_NEAR(imageParData[i*imageHeight+j],imageSeqData[i*imageHeight+j], 3e-2);

    wbExport("../dataset/outputtest.ppm", outputImagePar);
}