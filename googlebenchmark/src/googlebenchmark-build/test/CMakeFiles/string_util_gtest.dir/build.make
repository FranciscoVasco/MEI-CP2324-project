# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/275/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /snap/clion/275/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build

# Include any dependencies generated for this target.
include test/CMakeFiles/string_util_gtest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/string_util_gtest.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/string_util_gtest.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/string_util_gtest.dir/flags.make

test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o: test/CMakeFiles/string_util_gtest.dir/flags.make
test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o: /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/string_util_gtest.cc
test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o: test/CMakeFiles/string_util_gtest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o -MF CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o.d -o CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o -c /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/string_util_gtest.cc

test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.i"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/string_util_gtest.cc > CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.i

test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.s"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/string_util_gtest.cc -o CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.s

# Object files for target string_util_gtest
string_util_gtest_OBJECTS = \
"CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o"

# External object files for target string_util_gtest
string_util_gtest_EXTERNAL_OBJECTS =

test/string_util_gtest: test/CMakeFiles/string_util_gtest.dir/string_util_gtest.cc.o
test/string_util_gtest: test/CMakeFiles/string_util_gtest.dir/build.make
test/string_util_gtest: src/libbenchmark.a
test/string_util_gtest: lib/libgmock_main.a
test/string_util_gtest: lib/libgmock.a
test/string_util_gtest: lib/libgtest.a
test/string_util_gtest: test/CMakeFiles/string_util_gtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable string_util_gtest"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/string_util_gtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/string_util_gtest.dir/build: test/string_util_gtest
.PHONY : test/CMakeFiles/string_util_gtest.dir/build

test/CMakeFiles/string_util_gtest.dir/clean:
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && $(CMAKE_COMMAND) -P CMakeFiles/string_util_gtest.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/string_util_gtest.dir/clean

test/CMakeFiles/string_util_gtest.dir/depend:
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test/CMakeFiles/string_util_gtest.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : test/CMakeFiles/string_util_gtest.dir/depend

