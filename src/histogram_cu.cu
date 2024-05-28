#include "histogram_cu.cuh"

namespace cuda {
    constexpr auto THREADS_PER_BLOCK = 256;
    constexpr auto HISTOGRAM_LENGTH = 256;
    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    __global__ void calcOutput(float* output, unsigned char* image, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = static_cast<float>(image[idx]) / 255.0f;
        }
    }

    __global__ void calculate_image(float *data, unsigned char* image, int size, int *hist){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            auto r = (unsigned char) (255 * data[3*idx]);
            auto g = (unsigned char) (255 * data[3*idx + 1]);
            auto b = (unsigned char) (255 * data[3*idx + 2]);
            image[3*idx] = r;
            image[3*idx +1] = g;
            image[3*idx +2] = b;
            auto value = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            atomicAdd(&hist[value], 1);
        }
    }

    __global__ void fill_with_correct_color(unsigned char *image, float *cdf, float cdf_min, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            auto x = static_cast<unsigned char>(255 * (cdf[image[idx]] - cdf_min) / (1 - cdf_min));
            if(x > 255){
                image[idx] = static_cast<unsigned char>(255);
            }else if (x<0){
                image[idx] = static_cast<unsigned char>(0);
            }else{
                image[idx] = x;
            }
        }
    }

    void fill_cdf(int size,float *cdf, int *histogram){
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
    }


    void histogram_equalization(int width, int height, float *data,
                                float *output_image_data,
                                unsigned char *image,
                                int (&hist)[HISTOGRAM_LENGTH],
                                float (&cdf)[HISTOGRAM_LENGTH]){
        int dataSize = width * height;
        for (int & i : hist) {
            i = 0;
        }

        int gridSize;

        ///CREATE UNSIGNED CHAR IMAGE and hist
        float *d_data_image;
        unsigned char *d_image;
        int *d_hist;
        cudaMalloc((void**)&d_data_image, dataSize * 3 * sizeof(float));
        cudaMalloc((void**)&d_image, dataSize * 3 * sizeof(unsigned char));
        cudaMalloc((void**)&d_hist, HISTOGRAM_LENGTH * sizeof(int));
        cudaMemcpy(d_data_image, data, dataSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
        gridSize = (dataSize*3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        calculate_image<<<gridSize, THREADS_PER_BLOCK>>>(d_data_image, d_image, dataSize,d_hist);
        cudaMemcpy(image, d_image, dataSize * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(hist, d_hist, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);

        cdf[0] = prob(hist[0], dataSize);
        float cdf_min = cdf[0];
        fill_cdf(dataSize,cdf,hist);


        ///CALCULATE IMAGE FROM CDF
        unsigned char* imaged;
        float *dcdf;
        cudaMalloc((void **) &imaged, dataSize * 3 * sizeof(unsigned char));
        cudaMalloc((void**) &dcdf, HISTOGRAM_LENGTH * sizeof(float) );
        cudaMemcpy(imaged, image, dataSize*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(dcdf, cdf, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
        gridSize = (3*dataSize+ THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fill_with_correct_color<<<gridSize, THREADS_PER_BLOCK>>>(imaged,dcdf,cdf_min,dataSize*3);
        cudaMemcpy(image, imaged,  dataSize*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);

        ///CALCULATE OUTPUT
        float *doutput;
        unsigned char *image_data;
        cudaMalloc((void **) &doutput, dataSize*3* sizeof(float));
        cudaMalloc((void **) &image_data, dataSize*3*  sizeof(unsigned char));
        cudaMemcpy(doutput,output_image_data, dataSize*3*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(image_data,image, dataSize*3*sizeof(unsigned char),cudaMemcpyHostToDevice);
        calcOutput<<<gridSize, THREADS_PER_BLOCK>>>(doutput,image_data,dataSize*3);
        cudaMemcpy(output_image_data, doutput, dataSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);
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
        for(int i = 0 ; i< iterations;i++){
            histogram_equalization(width, height,
                                   data,output_image_data,
                                   uchar_image_arr,
                                   histogram,cdf);
            data = output_image_data;
        }
        return output_image;

    }


}