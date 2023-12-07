set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

set(TEST_FILES
	unit_linscan.cpp
	#perf_linscan.cpp
	max_sketch.cpp
	read_fvecs.cpp
	read_csr.cpp
	write_csr.cpp
	read_gt.cpp
	profile_sketch.cpp
	comp_groundtruth.cpp
	high_freq_record.cpp
	high_freq_convert.cpp
	trec_sample.cpp
	unit_parlay.cpp
	reorder.cpp
	#unit_reorder.cpp
)

foreach(TEST_FILE ${TEST_FILES})
	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
	add_executable(${TEST_NAME} ${TEST_FILE})
	target_link_libraries(${TEST_NAME} SparseANNLibrary OpenSSL::SSL OpenSSL::Crypto Parlay::parlay)
endforeach()
