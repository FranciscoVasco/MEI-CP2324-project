# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/src"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/build"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/tmp"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/stamp"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/download"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/third_party/googletest/stamp${cfgdir}") # cfgdir has leading slash
endif()
