
#include "histogram_par_f.h"
#include <omp.h>

namespace cp_par_f {
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

    void calculate_rgb(const float *input_image_data, unsigned char *uchar_image_arr, int *histogram, int size, int channels) {
        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);

        #pragma omp parallel for reduction(+:histogram[:HISTOGRAM_LENGTH])
        for (int i = 0; i < size; i++) {
            auto r = static_cast<unsigned char>(255 * input_image_data[channels * i]);
            auto g = static_cast<unsigned char>(255 * input_image_data[channels * i + 1]);
            auto b = static_cast<unsigned char>(255 * input_image_data[channels * i + 2]);
            uchar_image_arr[channels * i] = r;
            uchar_image_arr[channels * i + 1] = g;
            uchar_image_arr[channels * i + 2] = b;
            auto gray = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            histogram[gray]++;
        }
    }

    void calculate_cdf(const int *histogram, float *cdf, int size) {
        cdf[0] = prob(histogram[0], size);

        for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
        }
    }

    void apply_cdf(const unsigned char *uchar_image_arr, float *output_image_data, const float *cdf, float cdf_min, int size_channels) {
        #pragma omp parallel
        for (int i = 0; i < size_channels; i++) {
            auto corrected = correct_color(cdf[uchar_image_arr[i]], cdf_min);
            output_image_data[i] = static_cast<float>(corrected) / 255.0f;
        }
    }

    void histogram_equalization(const int width, const int height,
                                const float *input_image_data,
                                float *output_image_data,
                                unsigned char *uchar_image_arr,
                                int *histogram,
                                float *cdf) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        calculate_rgb(input_image_data, uchar_image_arr, histogram, size, channels);
        calculate_cdf(histogram, cdf, size);
        float cdf_min = cdf[0];
        apply_cdf(uchar_image_arr, output_image_data, cdf, cdf_min, size_channels);
    }

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations,
                                               bool parallel_histogram = true,
                                               bool parallel_cdf = true,
                                               bool parallel_apply = true) {

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

        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height, input_image_data, output_image_data,
                                   uchar_image_arr, histogram, cdf,
                                   parallel_histogram, parallel_cdf, parallel_apply);

            input_image_data = output_image_data;
        }

        delete[] uchar_image_arr;
        return output_image;
    }
}