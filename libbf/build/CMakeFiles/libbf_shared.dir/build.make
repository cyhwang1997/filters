# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/libbf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/libbf/build

# Include any dependencies generated for this target.
include CMakeFiles/libbf_shared.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/libbf_shared.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libbf_shared.dir/flags.make

CMakeFiles/libbf_shared.dir/src/bitvector.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/bitvector.cpp.o: ../src/bitvector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/libbf_shared.dir/src/bitvector.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/bitvector.cpp.o -c /home/ubuntu/libbf/src/bitvector.cpp

CMakeFiles/libbf_shared.dir/src/bitvector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/bitvector.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/bitvector.cpp > CMakeFiles/libbf_shared.dir/src/bitvector.cpp.i

CMakeFiles/libbf_shared.dir/src/bitvector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/bitvector.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/bitvector.cpp -o CMakeFiles/libbf_shared.dir/src/bitvector.cpp.s

CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.o: ../src/counter_vector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.o -c /home/ubuntu/libbf/src/counter_vector.cpp

CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/counter_vector.cpp > CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.i

CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/counter_vector.cpp -o CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.s

CMakeFiles/libbf_shared.dir/src/hash.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/hash.cpp.o: ../src/hash.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/libbf_shared.dir/src/hash.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/hash.cpp.o -c /home/ubuntu/libbf/src/hash.cpp

CMakeFiles/libbf_shared.dir/src/hash.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/hash.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/hash.cpp > CMakeFiles/libbf_shared.dir/src/hash.cpp.i

CMakeFiles/libbf_shared.dir/src/hash.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/hash.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/hash.cpp -o CMakeFiles/libbf_shared.dir/src/hash.cpp.s

CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.o: ../src/bloom_filter/a2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.o -c /home/ubuntu/libbf/src/bloom_filter/a2.cpp

CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/bloom_filter/a2.cpp > CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.i

CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/bloom_filter/a2.cpp -o CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.s

CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.o: ../src/bloom_filter/basic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.o -c /home/ubuntu/libbf/src/bloom_filter/basic.cpp

CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/bloom_filter/basic.cpp > CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.i

CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/bloom_filter/basic.cpp -o CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.s

CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.o: ../src/bloom_filter/bitwise.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.o -c /home/ubuntu/libbf/src/bloom_filter/bitwise.cpp

CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/bloom_filter/bitwise.cpp > CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.i

CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/bloom_filter/bitwise.cpp -o CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.s

CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.o: ../src/bloom_filter/counting.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.o -c /home/ubuntu/libbf/src/bloom_filter/counting.cpp

CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/bloom_filter/counting.cpp > CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.i

CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/bloom_filter/counting.cpp -o CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.s

CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.o: CMakeFiles/libbf_shared.dir/flags.make
CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.o: ../src/bloom_filter/stable.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.o -c /home/ubuntu/libbf/src/bloom_filter/stable.cpp

CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/libbf/src/bloom_filter/stable.cpp > CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.i

CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/libbf/src/bloom_filter/stable.cpp -o CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.s

# Object files for target libbf_shared
libbf_shared_OBJECTS = \
"CMakeFiles/libbf_shared.dir/src/bitvector.cpp.o" \
"CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.o" \
"CMakeFiles/libbf_shared.dir/src/hash.cpp.o" \
"CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.o" \
"CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.o" \
"CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.o" \
"CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.o" \
"CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.o"

# External object files for target libbf_shared
libbf_shared_EXTERNAL_OBJECTS =

lib/libbf.so: CMakeFiles/libbf_shared.dir/src/bitvector.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/src/counter_vector.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/src/hash.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/src/bloom_filter/a2.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/src/bloom_filter/basic.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/src/bloom_filter/bitwise.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/src/bloom_filter/counting.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/src/bloom_filter/stable.cpp.o
lib/libbf.so: CMakeFiles/libbf_shared.dir/build.make
lib/libbf.so: CMakeFiles/libbf_shared.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/libbf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX shared library lib/libbf.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libbf_shared.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libbf_shared.dir/build: lib/libbf.so

.PHONY : CMakeFiles/libbf_shared.dir/build

CMakeFiles/libbf_shared.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libbf_shared.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libbf_shared.dir/clean

CMakeFiles/libbf_shared.dir/depend:
	cd /home/ubuntu/libbf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/libbf /home/ubuntu/libbf /home/ubuntu/libbf/build /home/ubuntu/libbf/build /home/ubuntu/libbf/build/CMakeFiles/libbf_shared.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libbf_shared.dir/depend

