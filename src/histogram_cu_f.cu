#include "histogram_cu_f.cuh"

namespace cuda_f {
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

    void calculate_image_wrap(float* d_data_image, unsigned char* d_image, int* d_hist, int dataSize) {
        int gridSize = (dataSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        calculate_image<<<gridSize, THREADS_PER_BLOCK>>>(d_data_image, d_image, dataSize, d_hist);
        cudaDeviceSynchronize();
    }

    __global__ void fill_with_correct_color(unsigned char *image, const float *cdf, float cdf_min, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            auto x = static_cast<unsigned char>(255 * (cdf[image[idx]] - cdf_min) / (1 - cdf_min));
            if(x > 255){
                image[idx] = static_cast<unsigned char>(255);
            }else if (x < 0){
                image[idx] = static_cast<unsigned char>(0);
            }else{
                image[idx] = x;
            }
        }
    }

    void fill_with_correct_color_wrap(unsigned char* d_image, float* dcdf, float cdf_min, int dataSize) {
        int gridSize = (3 * dataSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fill_with_correct_color<<<gridSize, THREADS_PER_BLOCK>>>(d_image, dcdf, cdf_min, dataSize * 3);
        cudaDeviceSynchronize();
    }

    void fill_cdf(int size, float *cdf, int *histogram){
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
    }

    void alloc_memory(float*& d_data_image, unsigned char*& d_image, int*& d_hist, int dataSize) {
        cudaMalloc((void**)&d_data_image, dataSize * 3 * sizeof(float));
        cudaMalloc((void**)&d_image, dataSize * 3 * sizeof(unsigned char));
        cudaMalloc((void**)&d_hist, HISTOGRAM_LENGTH * sizeof(int));
    }

    void host_to_device(float* d_data_image, int* d_hist, const float* data, const int* hist, int dataSize) {
        cudaMemcpy(d_data_image, data, dataSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hist, hist, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
    }

    void copy_hist_to_host(int* hist, int* d_hist) {
        cudaMemcpy(hist, d_hist, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    }

    void calc_output_wrap(float* doutput, unsigned char* d_image, int dataSize) {
        int gridSize = (3 * dataSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        calcOutput<<<gridSize, THREADS_PER_BLOCK>>>(doutput, d_image, dataSize * 3);
        cudaDeviceSynchronize();
    }

    void histogram_equalization(int width, int height, float *data,
                                float *output_image_data,
                                int (&hist)[HISTOGRAM_LENGTH],
                                float (&cdf)[HISTOGRAM_LENGTH]){
        int dataSize = width * height;
        for (int & i : hist) {
            i = 0;
        }

        float *d_data_image;
        unsigned char *d_image;
        int *d_hist;
        alloc_memory(d_data_image, d_image, d_hist, dataSize);
        host_to_device(d_data_image, d_hist, data, hist, dataSize);

        calculate_image_wrap(d_data_image, d_image, d_hist, dataSize);
        copy_hist_to_host(hist, d_hist);

        cdf[0] = prob(hist[0], dataSize);
        float cdf_min = cdf[0];
        fill_cdf(dataSize, cdf, hist);

        float *dcdf;
        cudaMalloc((void**) &dcdf, HISTOGRAM_LENGTH * sizeof(float));
        cudaMemcpy(dcdf, cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
        fill_with_correct_color_wrap(d_image, dcdf, cdf_min, dataSize);

        float *doutput;
        cudaMalloc((void**) &doutput, dataSize * 3 * sizeof(float));
        cudaMemcpy(doutput, output_image_data, dataSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
        calc_output_wrap(doutput, d_image, dataSize);
        cudaMemcpy(output_image_data, doutput, dataSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_data_image);
        cudaFree(d_image);
        cudaFree(d_hist);
        cudaFree(dcdf);
        cudaFree(doutput);
    }

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        float *data = wbImage_getData(input_image);
        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        wbImage_t output_image = wbImage_new(width, height, 3);
        float *output_image_data = wbImage_getData(output_image);
        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        for(int i = 0; i < iterations; i++){
            histogram_equalization(width, height, data, output_image_data, histogram, cdf);
            data = output_image_data;
        }
        return output_image;
    }
}
