# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

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

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.17.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.17.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build"

# Include any dependencies generated for this target.
include CMakeFiles/tp2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tp2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tp2.dir/flags.make

CMakeFiles/tp2.dir/src/main.cpp.o: CMakeFiles/tp2.dir/flags.make
CMakeFiles/tp2.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tp2.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tp2.dir/src/main.cpp.o -c "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/main.cpp"

CMakeFiles/tp2.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tp2.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/main.cpp" > CMakeFiles/tp2.dir/src/main.cpp.i

CMakeFiles/tp2.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tp2.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/main.cpp" -o CMakeFiles/tp2.dir/src/main.cpp.s

CMakeFiles/tp2.dir/src/knn.cpp.o: CMakeFiles/tp2.dir/flags.make
CMakeFiles/tp2.dir/src/knn.cpp.o: ../src/knn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tp2.dir/src/knn.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tp2.dir/src/knn.cpp.o -c "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/knn.cpp"

CMakeFiles/tp2.dir/src/knn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tp2.dir/src/knn.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/knn.cpp" > CMakeFiles/tp2.dir/src/knn.cpp.i

CMakeFiles/tp2.dir/src/knn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tp2.dir/src/knn.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/knn.cpp" -o CMakeFiles/tp2.dir/src/knn.cpp.s

CMakeFiles/tp2.dir/src/pca.cpp.o: CMakeFiles/tp2.dir/flags.make
CMakeFiles/tp2.dir/src/pca.cpp.o: ../src/pca.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/tp2.dir/src/pca.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tp2.dir/src/pca.cpp.o -c "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/pca.cpp"

CMakeFiles/tp2.dir/src/pca.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tp2.dir/src/pca.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/pca.cpp" > CMakeFiles/tp2.dir/src/pca.cpp.i

CMakeFiles/tp2.dir/src/pca.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tp2.dir/src/pca.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/pca.cpp" -o CMakeFiles/tp2.dir/src/pca.cpp.s

CMakeFiles/tp2.dir/src/eigen.cpp.o: CMakeFiles/tp2.dir/flags.make
CMakeFiles/tp2.dir/src/eigen.cpp.o: ../src/eigen.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/tp2.dir/src/eigen.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tp2.dir/src/eigen.cpp.o -c "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/eigen.cpp"

CMakeFiles/tp2.dir/src/eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tp2.dir/src/eigen.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/eigen.cpp" > CMakeFiles/tp2.dir/src/eigen.cpp.i

CMakeFiles/tp2.dir/src/eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tp2.dir/src/eigen.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/src/eigen.cpp" -o CMakeFiles/tp2.dir/src/eigen.cpp.s

# Object files for target tp2
tp2_OBJECTS = \
"CMakeFiles/tp2.dir/src/main.cpp.o" \
"CMakeFiles/tp2.dir/src/knn.cpp.o" \
"CMakeFiles/tp2.dir/src/pca.cpp.o" \
"CMakeFiles/tp2.dir/src/eigen.cpp.o"

# External object files for target tp2
tp2_EXTERNAL_OBJECTS =

tp2: CMakeFiles/tp2.dir/src/main.cpp.o
tp2: CMakeFiles/tp2.dir/src/knn.cpp.o
tp2: CMakeFiles/tp2.dir/src/pca.cpp.o
tp2: CMakeFiles/tp2.dir/src/eigen.cpp.o
tp2: CMakeFiles/tp2.dir/build.make
tp2: CMakeFiles/tp2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable tp2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tp2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tp2.dir/build: tp2

.PHONY : CMakeFiles/tp2.dir/build

CMakeFiles/tp2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tp2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tp2.dir/clean

CMakeFiles/tp2.dir/depend:
	cd "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2" "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2" "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build" "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build" "/Users/desarrollo/Metodos Numericos/TP2/metnum-tp2/build/CMakeFiles/tp2.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/tp2.dir/depend

