# CMake generated Testfile for 
# Source directory: /home/ubuntu/filters/libbf/test
# Build directory: /home/ubuntu/filters/libbf/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unit "/home/ubuntu/filters/libbf/build/bin/bf-test")
set_tests_properties(unit PROPERTIES  _BACKTRACE_TRIPLES "/home/ubuntu/filters/libbf/test/CMakeLists.txt;6;add_test;/home/ubuntu/filters/libbf/test/CMakeLists.txt;0;")
subdirs("bf")
