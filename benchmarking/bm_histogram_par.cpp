#include "bm_histogram_par.h"
#include "benchmark/benchmark.h"
#include "histogram_par.h"
#include "histogram_eq.h"
#include "histogram_cu.cuh"
#include <wb.h>

#define DATASET_FOLDER "../dataset/"

wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");

static void BM_Par_Hist(benchmark::State& state){
    for (auto _ : state) {
        cp_par::iterative_histogram_equalization(inputImage, state.range(0));
    }
}

static void BM_Seq_Hist(benchmark::State& state){
    for (auto _ : state) {
        cp::iterative_histogram_equalization(inputImage, state.range(0));
    }
}

static void BM_CUDA_Hist(benchmark::State& state){
    for (auto _ : state) {
        cuda::iterative_histogram_equalization(inputImage, state.range(0));
    }
}

BENCHMARK(BM_Par_Hist)
->Arg(1)
->Arg(5)
->Arg(10);
BENCHMARK(BM_Seq_Hist)
->Arg(1)
->Arg(5)
->Arg(10);
BENCHMARK(BM_CUDA_Hist)
->Arg(1)
->Arg(5)
->Arg(10);


BENCHMARK_MAIN();