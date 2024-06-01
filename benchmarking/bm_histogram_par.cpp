#include "benchmark/benchmark.h"
#include "histogram_par.h"
#include "histogram_eq.h"
#include "histogram_cu.cuh"
#include <wb.h>
#include <chrono>
#include <thread>
#define DATASET_FOLDER "../dataset/"

wbImage_t inputImage512 = wbImport(DATASET_FOLDER "image512.ppm");
wbImage_t inputImage1024 = wbImport(DATASET_FOLDER "image1024.ppm");
wbImage_t inputImage2048 = wbImport(DATASET_FOLDER "image2048.ppm");
wbImage_t inputImage4096 = wbImport(DATASET_FOLDER "image4096.ppm");

static void BM_Par_Hist512(benchmark::State& state){
    for (auto _ : state) {

        auto image = cp_par::iterative_histogram_equalization(inputImage512, state.range(0));
        wbImage_delete(image);

    }
}

static void BM_Seq_Hist512(benchmark::State& state){
    for (auto _ : state) {
        auto image = cp::iterative_histogram_equalization(inputImage512, state.range(0));
        wbImage_delete(image);
    }
}

static void BM_CUDA_Hist512(benchmark::State& state){
    for (auto _ : state) {
        auto now = std::chrono::system_clock::now();
        auto image = cuda::iterative_histogram_equalization(inputImage512, state.range(0));
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end - now;
        state.SetIterationTime(duration.count());
        wbImage_delete(image);
    }
}

static void BM_Par_Hist1024(benchmark::State& state){
    for (auto _ : state) {

        auto image = cp_par::iterative_histogram_equalization(inputImage1024, state.range(0));

        wbImage_delete(image);
    }
}

static void BM_Seq_Hist1024(benchmark::State& state){
    for (auto _ : state) {

        auto image = cp::iterative_histogram_equalization(inputImage1024, state.range(0));

        wbImage_delete(image);
    }
}

static void BM_CUDA_Hist1024(benchmark::State& state){
    for (auto _ : state) {
        auto now = std::chrono::system_clock::now();
        auto image = cuda::iterative_histogram_equalization(inputImage1024, state.range(0));
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end - now;
        state.SetIterationTime(duration.count());
        
        wbImage_delete(image);
    }
}

static void BM_Par_Hist2048(benchmark::State& state){
    for (auto _ : state) {

        auto image = cp_par::iterative_histogram_equalization(inputImage2048, state.range(0));

        
        wbImage_delete(image);
    }
}

static void BM_Seq_Hist2048(benchmark::State& state){
    for (auto _ : state) {

        auto image = cp::iterative_histogram_equalization(inputImage2048, state.range(0));

        
        wbImage_delete(image);
    }
}

static void BM_CUDA_Hist2048(benchmark::State& state){
    for (auto _ : state) {
        auto now = std::chrono::system_clock::now();
        auto image = cuda::iterative_histogram_equalization(inputImage2048, state.range(0));
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end - now;
        state.SetIterationTime(duration.count());
        
        wbImage_delete(image);
    }
}

static void BM_Par_Hist4096(benchmark::State& state){
    for (auto _ : state) {

        auto image = cp_par::iterative_histogram_equalization(inputImage4096, state.range(0));

        wbImage_delete(image);
    }
}

static void BM_Seq_Hist4096(benchmark::State& state){
    for (auto _ : state) {

        auto image = cp::iterative_histogram_equalization(inputImage4096, state.range(0));

        wbImage_delete(image);
    }
}

static void BM_CUDA_Hist4096(benchmark::State& state){
    for (auto _ : state) {
        auto now = std::chrono::system_clock::now();
        auto image = cuda::iterative_histogram_equalization(inputImage4096, state.range(0));
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end - now;
        state.SetIterationTime(duration.count());
        wbImage_delete(image);
    }
}

BENCHMARK(BM_Par_Hist512)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Seq_Hist512)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CUDA_Hist512)
->Arg(1)->UseManualTime()
->Arg(5)->UseManualTime()
->Arg(10)->UseManualTime()
->Arg(20)->UseManualTime()
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Par_Hist1024)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Seq_Hist1024)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CUDA_Hist1024)
->Arg(1)->UseManualTime()
->Arg(5)->UseManualTime()
->Arg(10)->UseManualTime()
->Arg(20)->UseManualTime()
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Par_Hist2048)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Seq_Hist2048)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CUDA_Hist2048)
->Arg(1)->UseManualTime()
->Arg(5)->UseManualTime()
->Arg(10)->UseManualTime()
->Arg(20)->UseManualTime()
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Par_Hist4096)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Seq_Hist4096)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20)
->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CUDA_Hist4096)
->Arg(1)->UseManualTime()
->Arg(5)->UseManualTime()
->Arg(10)->UseManualTime()
->Arg(20)->UseManualTime()
->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
