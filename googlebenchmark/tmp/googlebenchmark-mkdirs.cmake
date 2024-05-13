# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/tmp"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-stamp"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src"
  "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-stamp${cfgdir}") # cfgdir has leading slash
endif()
