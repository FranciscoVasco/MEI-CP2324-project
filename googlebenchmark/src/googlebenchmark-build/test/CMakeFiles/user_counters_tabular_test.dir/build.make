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
include test/CMakeFiles/user_counters_tabular_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/user_counters_tabular_test.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/user_counters_tabular_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/user_counters_tabular_test.dir/flags.make

test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o: test/CMakeFiles/user_counters_tabular_test.dir/flags.make
test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o: /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/user_counters_tabular_test.cc
test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o: test/CMakeFiles/user_counters_tabular_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o -MF CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o.d -o CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o -c /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/user_counters_tabular_test.cc

test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.i"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/user_counters_tabular_test.cc > CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.i

test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.s"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test/user_counters_tabular_test.cc -o CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.s

# Object files for target user_counters_tabular_test
user_counters_tabular_test_OBJECTS = \
"CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o"

# External object files for target user_counters_tabular_test
user_counters_tabular_test_EXTERNAL_OBJECTS =

test/user_counters_tabular_test: test/CMakeFiles/user_counters_tabular_test.dir/user_counters_tabular_test.cc.o
test/user_counters_tabular_test: test/CMakeFiles/user_counters_tabular_test.dir/build.make
test/user_counters_tabular_test: test/liboutput_test_helper.a
test/user_counters_tabular_test: src/libbenchmark_main.a
test/user_counters_tabular_test: src/libbenchmark.a
test/user_counters_tabular_test: test/CMakeFiles/user_counters_tabular_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable user_counters_tabular_test"
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/user_counters_tabular_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/user_counters_tabular_test.dir/build: test/user_counters_tabular_test
.PHONY : test/CMakeFiles/user_counters_tabular_test.dir/build

test/CMakeFiles/user_counters_tabular_test.dir/clean:
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test && $(CMAKE_COMMAND) -P CMakeFiles/user_counters_tabular_test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/user_counters_tabular_test.dir/clean

test/CMakeFiles/user_counters_tabular_test.dir/depend:
	cd /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark/test /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test /home/tomedias/CLionProjects/MEI-CP2324-project/googlebenchmark/src/googlebenchmark-build/test/CMakeFiles/user_counters_tabular_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : test/CMakeFiles/user_counters_tabular_test.dir/depend

