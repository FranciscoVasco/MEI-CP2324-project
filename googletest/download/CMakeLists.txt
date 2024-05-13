cmake_minimum_required(VERSION 3.10)
project(googletest LANGUAGES C CXX)

include(ExternalProject)

set (GTEST_HOME /home/tomedias/CLionProjects/MEI-CP2324-project/../googletest)

ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
    SOURCE_DIR "/home/tomedias/CLionProjects/MEI-CP2324-project/googletest/src"
    BINARY_DIR "/home/tomedias/CLionProjects/MEI-CP2324-project/googletest/build"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)
