//
// Created by tomedias on 5/17/24.
//
#include "histogram_cu.cuh"

namespace cuda {
    constexpr auto THREADS_PER_BLOCK = 512;
    constexpr auto HISTOGRAM_LENGTH = 256;
    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }
    unsigned char inline clamp(unsigned char x) {
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }
    unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
    }
    ///DONE


    __global__ void histogramKernel(unsigned char *data, int *hist, int dataSize, int numBins) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < dataSize) {
            int bin = data[idx];
            if (bin < numBins) {
                atomicAdd(&hist[bin], 1);
            }
        }
    }


    __global__ void grayImage(unsigned char *image_data, unsigned char* grayImage, int dataSize){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < dataSize) {
            auto r = image_data[3*idx];
            auto g = image_data[3*idx+1];
            auto b = image_data[3 * idx + 2];
            grayImage[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }
    }

    __global__ void calcOutput(float* output, unsigned char* image, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = static_cast<float>(image[idx]) / 255.0f;
        }
    }

    __global__ void calculate_image(float *data, unsigned char* image, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            image[idx] = (unsigned char) (255 * data[idx]);
        }

    }






    //TODO

    void fill_with_correct_color(int size_channels,float cdf_min, unsigned char *uchar_image_arr, const float *cdf){

        for (int i = 0; i < size_channels; i++)
            uchar_image_arr[i] = correct_color(cdf[uchar_image_arr[i]], cdf_min);
    }
    void fill_cdf(int size,float *cdf, int *histogram){
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
    }


    void histogram_equalization(int width, int height, float *data,
                                float *output_image_data,
                                unsigned char *image,
                                unsigned char *gray_image,
                                int (&hist)[HISTOGRAM_LENGTH],
                                float (&cdf)[HISTOGRAM_LENGTH]){
        int dataSize = width * height;
        for (int & i : hist) {
            i = 0;
        }

        int gridSize;


        float *d_data_image;
        unsigned char *d_image;
        cudaMalloc((void**)&d_data_image, dataSize * 3 * sizeof(float));
        cudaMalloc((void**)&d_image, dataSize * 3 * sizeof(unsigned char));
        cudaMemcpy(d_data_image, data, dataSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
        gridSize = (dataSize*3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        calculate_image<<<gridSize, THREADS_PER_BLOCK>>>(d_data_image, d_image, dataSize * 3);
        cudaMemcpy(image, d_image, dataSize * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);



        ///CREATE GRAY IMAGE
        unsigned char *d_data;
        unsigned char*dgray_image;
        cudaMalloc((void**)&d_data, dataSize*3*sizeof(float));
        cudaMalloc((void **)&dgray_image, dataSize*sizeof(unsigned char));
        cudaMemcpy(d_data, image, dataSize*3*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(dgray_image,gray_image, dataSize * sizeof(unsigned char),cudaMemcpyHostToDevice);
        gridSize = (dataSize+ THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        grayImage<<<gridSize, THREADS_PER_BLOCK>>>(d_data, dgray_image, dataSize);
        cudaMemcpy(gray_image, dgray_image, dataSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);


        ///FILL HISTOGRAM
        unsigned char *gray_data;
        int *d_hist;
        cudaMalloc((void **) &gray_data, dataSize* sizeof(float));
        cudaMalloc((void **) &d_hist, HISTOGRAM_LENGTH * sizeof(int));
        cudaMemcpy(gray_data, gray_image, dataSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hist, hist, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
        histogramKernel<<<gridSize, THREADS_PER_BLOCK>>>(gray_data, d_hist, dataSize, HISTOGRAM_LENGTH);
        cudaMemcpy(hist, d_hist, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);


        cdf[0] = prob(hist[0], dataSize);
        float cdf_min = cdf[0];
        fill_cdf(dataSize,cdf,hist);
        fill_with_correct_color(dataSize*3,cdf_min,image,cdf);


        ///CALCULATE OUTPUT
        float *doutput;
        unsigned char *image_data;
        cudaMalloc((void **) &doutput, dataSize*3* sizeof(float));
        cudaMalloc((void **) &image_data, dataSize*3*  sizeof(unsigned char));
        cudaMemcpy(doutput,output_image_data, dataSize*3*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(image_data,image, dataSize*3*sizeof(unsigned char),cudaMemcpyHostToDevice);
        calcOutput<<<gridSize*3, THREADS_PER_BLOCK>>>(doutput,image_data,dataSize*3);
        cudaMemcpy(output_image_data, doutput, dataSize*3 * sizeof(float), cudaMemcpyDeviceToHost);

    }


    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        float *data = wbImage_getData(input_image);
        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        wbImage_t output_image = wbImage_new(width, height, 3);
        float *output_image_data = wbImage_getData(output_image);
        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];
        auto *uchar_image_arr = new unsigned char[width*height*3];
        auto *gray_image_arr = new unsigned char[width*height];
        for(int i = 0 ; i< iterations;i++){
            histogram_equalization(width, height,
                                   data,output_image_data,
                                   uchar_image_arr, gray_image_arr,
                                   histogram,cdf);
            data = output_image_data;
        }
        return output_image;

    }


}