//
// Created by herve on 13-04-2024.
//

#include "histogram_par.h"
#include <omp.h>

namespace cp_par {
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

    static void histogram_equalization(const int width, const int height,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       unsigned char *uchar_image_arr,
                                       unsigned char *gray_image_arr,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        omp_set_num_threads(4);

#pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            uchar_image_arr[i] = (unsigned char) (255 * input_image_data[i]);

#pragma omp parallel for
        for (int i = 0; i < size; i++){
            auto r = uchar_image_arr[3 * i];
            auto g = uchar_image_arr[3 * i + 1];
            auto b = uchar_image_arr[3 * i + 2];
            gray_image_arr[i] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
#pragma omp parallel for reduction(+:histogram[:HISTOGRAM_LENGTH])
        for (int i = 0; i < size; i++)
#pragma omp atomic
                histogram[gray_image_arr[i]]++;

        cdf[0] = prob(histogram[0], size);
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);

        auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);

#pragma omp parallel for
        for (int i = 0; i < size_channels; i++){
            uchar_image_arr[i] = correct_color(cdf[uchar_image_arr[i]], cdf_min);
            output_image_data[i] = static_cast<float>(uchar_image_arr[i]) / 255.0f;
        }

//        #pragma omp parallel for
//        for (int i = 0; i < size_channels; i++)
//            o
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



        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];
        auto *uchar_image_arr = new unsigned char[size_channels];
        auto *gray_image_arr = new unsigned char[size];
        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image_arr, gray_image_arr,
                                   histogram, cdf);

            input_image_data = output_image_data;
        }

        return output_image;
    }
}