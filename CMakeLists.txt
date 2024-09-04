cmake_minimum_required(VERSION 3.10)
project(SparseANN)

set(CMAKE_CXX_STANDARD 17)

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

include_directories(SparseParlayANN/algorithms/util)

add_subdirectory(src)
add_subdirectory(test)
