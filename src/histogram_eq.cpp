//
// Created by herve on 13-04-2024.
//

#include "histogram_par.h"
#include <omp.h>

namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256;

    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    static unsigned char inline clamp(unsigned char x) {
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }

    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
    }

    static void init_image_arr(int size_channels,unsigned char *uchar_image_arr, const float *input_image_data){

        #pragma omp parallel for

        for (int i = 0; i < size_channels; i++)
            uchar_image_arr[i] = (unsigned char) (255 * input_image_data[i]);
    }


    static void calculate_rgb(int height, int width, const unsigned char* uchar_image_arr,unsigned  char* gray_image_arr){

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                auto idx = i * width + j;
                auto r = uchar_image_arr[3 * idx];
                auto g = uchar_image_arr[3 * idx + 1];
                auto b = uchar_image_arr[3 * idx + 2];
                gray_image_arr[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }
    }

    static void fill_hist(int size, int *histogram,const unsigned char* gray_image_arr){

        for (int i = 0; i < size; i++)
            histogram[gray_image_arr[i]]++;
    }

    static void fill_cdf(int size,float *cdf, int *histogram){
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
    }

    static float calculate_cdf_min(const float *cdf){
        auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);
        return cdf_min;
    }

    static void fill_with_correct_color(int size_channels,float cdf_min, unsigned char *uchar_image_arr, const float *cdf){

        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            uchar_image_arr[i] = correct_color(cdf[uchar_image_arr[i]], cdf_min);
    }

    static void fill_output(int size_channels, float *output_image_data, const unsigned char *uchar_image_arr){

        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image_arr[i]) / 255.0f;
    }




    static void histogram_equalization(const int width, const int height,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       const std::shared_ptr<unsigned char[]> &uchar_image,
                                       const std::shared_ptr<unsigned char[]> &gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        unsigned char *uchar_image_arr = uchar_image.get();
        unsigned char *gray_image_arr = gray_image.get();
        init_image_arr(size_channels,uchar_image_arr,input_image_data);
        omp_set_num_threads(8); // nr of cores of the cpu
        calculate_rgb(height, width, uchar_image_arr,
                      gray_image_arr);

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);

        fill_hist(size,histogram,gray_image_arr);

        cdf[0] = prob(histogram[0], size);

        fill_cdf(size,cdf,histogram);


        float cdf_min = calculate_cdf_min(cdf);


        fill_with_correct_color(size_channels,cdf_min,uchar_image_arr,cdf);
        fill_output(size_channels,output_image_data,uchar_image_arr);
    }

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {

        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        wbImage_t output_image = wbImage_new(width, height, channels);
        float *input_image_data = wbImage_getData(input_image);
        float *output_image_data = wbImage_getData(output_image);

        std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
        std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf);

            input_image_data = output_image_data;
        }

        return output_image;
    }
}