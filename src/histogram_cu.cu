#include "histogram_cu.cuh"

namespace cuda {
    constexpr auto THREADS_PER_BLOCK = 256;
    constexpr auto HISTOGRAM_LENGTH = 256;
    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    __global__ void calcOutput(float* output, const unsigned char* image, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = static_cast<float>(image[idx]) / 255.0f;
        }
    }

    __global__ void calculate_image(const float *data, unsigned char* image, int size, int *hist){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            auto r = (unsigned char) (255 * data[3*idx]);
            auto g = (unsigned char) (255 * data[3*idx + 1]);
            auto b = (unsigned char) (255 * data[3*idx + 2]);
            image[3*idx] = r;
            image[3*idx + 1] = g;
            image[3*idx + 2] = b;
            auto value = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            atomicAdd(&hist[value], 1);

        }
    }

    __global__ void fill_with_correct_color(unsigned char *image,const float *cdf, float cdf_min, int size){
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
        cudaMalloc((void**)&d_image, dataSize * 3 * sizeof(float));
        cudaMalloc((void**)&d_hist, HISTOGRAM_LENGTH * sizeof(int));
        cudaMemcpy(d_data_image, data, dataSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hist, hist, HISTOGRAM_LENGTH*sizeof(int), cudaMemcpyHostToDevice);
        gridSize = (dataSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        calculate_image<<<gridSize, THREADS_PER_BLOCK>>>(d_data_image, d_image, dataSize,d_hist);
        cudaDeviceSynchronize();
        cudaMemcpy(hist, d_hist, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0 ; i< HISTOGRAM_LENGTH;i++){
//            std::cout << i << ": " << hist[i] <<  std::endl;
//        }

        cdf[0] = prob(hist[0], dataSize);
        float cdf_min = cdf[0];
        fill_cdf(dataSize,cdf,hist);

        ///CALCULATE IMAGE FROM CDF
        float *dcdf;
        cudaMalloc((void**) &dcdf, HISTOGRAM_LENGTH * sizeof(float) );
        cudaMemcpy(dcdf, cdf, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
        gridSize = (3*dataSize+ THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fill_with_correct_color<<<gridSize, THREADS_PER_BLOCK>>>(d_image,dcdf,cdf_min,dataSize*3);
        cudaDeviceSynchronize();
        ///CALCULATE OUTPUT
        float *doutput;
        cudaMalloc((void **) &doutput, dataSize*3* sizeof(float));
        cudaMemcpy(doutput,output_image_data, dataSize*3*sizeof(float),cudaMemcpyHostToDevice);
        calcOutput<<<gridSize, THREADS_PER_BLOCK>>>(doutput,d_image,dataSize*3);
        cudaDeviceSynchronize();
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
                                   histogram,cdf);
            data = output_image_data;
        }
        return output_image;

    }


}