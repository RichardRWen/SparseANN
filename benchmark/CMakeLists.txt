set(BENCHMARK_FILES
	parlayann_unit_test.cpp
)

foreach(BENCHMARK_FILE ${BENCHMARK_FILES})
	get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
	add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILE})
	target_link_libraries(${BENCHMARK_NAME} OpenSSL::SSL OpenSSL::Crypto)
    #target_compile_options(${TEST_NAME} PRIVATE -mavx2)
endforeach()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/parlaylib/include)
include_directories(${CMAKE_SOURCE_DIR}/SparseParlayANN/algorithms)
