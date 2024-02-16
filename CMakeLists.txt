cmake_minimum_required(VERSION 3.10)
project(SparseANN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH "parlaylib/lib")

find_package(Parlay REQUIRED)
set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

include_directories(include)
include_directories(SparseParlayANN/algorithms/util)

add_subdirectory(src)
add_subdirectory(test)
