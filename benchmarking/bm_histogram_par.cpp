#include "bm_histogram_par.h"
#include "benchmark/benchmark.h"
#include "histogram_par.h"
#include "histogram_eq.h"
#include "histogram_cu.cuh"
#include <wb.h>

#define DATASET_FOLDER "../dataset/"

wbImage_t inputImage512 = wbImport(DATASET_FOLDER "image512.ppm");
wbImage_t inputImage1024 = wbImport(DATASET_FOLDER "image1024.ppm");
wbImage_t inputImage2048 = wbImport(DATASET_FOLDER "image2048.ppm");
wbImage_t inputImage4096 = wbImport(DATASET_FOLDER "input4096.ppm");

static void BM_Par_Hist512(benchmark::State& state){
    for (auto _ : state) {
        cp_par::iterative_histogram_equalization(inputImage512, state.range(0));
    }
}

static void BM_Seq_Hist512(benchmark::State& state){
    for (auto _ : state) {
        cp::iterative_histogram_equalization(inputImage512, state.range(0));
    }
}

static void BM_CUDA_Hist512(benchmark::State& state){
    for (auto _ : state) {
        cuda::iterative_histogram_equalization(inputImage512, state.range(0));
    }
}

static void BM_Par_Hist1024(benchmark::State& state){
    for (auto _ : state) {
        cp_par::iterative_histogram_equalization(inputImage1024, state.range(0));
    }
}

static void BM_Seq_Hist1024(benchmark::State& state){
    for (auto _ : state) {
        cp::iterative_histogram_equalization(inputImage1024, state.range(0));
    }
}

static void BM_CUDA_Hist1024(benchmark::State& state){
    for (auto _ : state) {
        cuda::iterative_histogram_equalization(inputImage1024, state.range(0));
    }
}

static void BM_Par_Hist2048(benchmark::State& state){
    for (auto _ : state) {
        cp_par::iterative_histogram_equalization(inputImage2048, state.range(0));
    }
}

static void BM_Seq_Hist2048(benchmark::State& state){
    for (auto _ : state) {
        cp::iterative_histogram_equalization(inputImage2048, state.range(0));
    }
}

static void BM_CUDA_Hist2048(benchmark::State& state){
    for (auto _ : state) {
        cuda::iterative_histogram_equalization(inputImage2048, state.range(0));
    }
}

static void BM_Par_Hist4096(benchmark::State& state){
    for (auto _ : state) {
        cp_par::iterative_histogram_equalization(inputImage4096, state.range(0));
    }
}

static void BM_Seq_Hist4096(benchmark::State& state){
    for (auto _ : state) {
        cp::iterative_histogram_equalization(inputImage4096, state.range(0));
    }
}

static void BM_CUDA_Hist4096(benchmark::State& state){
    for (auto _ : state) {
        cuda::iterative_histogram_equalization(inputImage4096, state.range(0));
    }
}

BENCHMARK(BM_Par_Hist512)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_Seq_Hist512)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_CUDA_Hist512)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_Par_Hist1024)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_Seq_Hist1024)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_CUDA_Hist1024)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_Par_Hist2048)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_Seq_Hist2048)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_CUDA_Hist2048)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_Par_Hist4096)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_Seq_Hist4096)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);
BENCHMARK(BM_CUDA_Hist4096)
->Arg(1)
->Arg(5)
->Arg(10)
->Arg(20);

BENCHMARK_MAIN();