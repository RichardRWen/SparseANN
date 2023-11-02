cmake_minimum_required(VERSION 3.10)
project(SparseANN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH "/home/richard/parlaylib/lib")

find_package(Parlay REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
