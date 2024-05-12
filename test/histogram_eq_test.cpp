#include <filesystem>
#include "gtest/gtest.h"
#include "histogram_par.h"
#include "histogram_eq.h"
#include <wb.h>

//using namespace cp_par;

#define DATASET_FOLDER "../dataset/"

TEST(HistogramEq, Input01_4) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");
    wbImage_t outputImagePar = cp_par::iterative_histogram_equalization(inputImage, 4);
    wbImage_t outputImageSeq = cp::iterative_histogram_equalization(inputImage, 4);

    const int imageWidth = wbImage_getWidth(outputImagePar);
    const int imageHeight = wbImage_getHeight(outputImagePar);

    float* imageParData = wbImage_getData(outputImagePar);
    float* imageSeqData = wbImage_getData(outputImageSeq);

    for(int i = 0; i < imageWidth; i++)
        for(int j = 0; j < imageHeight; j++)
            EXPECT_EQ(imageParData[i*imageHeight+j],imageSeqData[i*imageHeight+j]);


    // check if the output image is correct
}