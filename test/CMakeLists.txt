set(TEST_FILES
	unit_linscan.cpp
	#perf_linscan.cpp
	max_sketch.cpp
	near_sketch.cpp
    count_min_sketch.cpp
	read_fvecs.cpp
	read_csr.cpp
	write_csr.cpp
	csr_to_fvecs.cpp
	read_gt.cpp
	profile_sketch.cpp
	comp_csr_groundtruth.cpp
	comp_bin_groundtruth.cpp
	measure_recall.cpp
	measure_overretrieval.cpp
    benchmark_bitvector.cpp
	generate_dataset.cpp
	high_freq_record.cpp
	high_freq_convert.cpp
	trec_sample.cpp
    truncate_csr.cpp
	unit_parlay.cpp
	reorder.cpp
	unit_reorder.cpp
	apply_ordering.cpp
	mips_histogram.cpp
	compare_dist.cpp
    eval_dist_approx.cpp
    graph_bfs_tree.cpp
    graph_bfs_tree_donut.cpp
)

foreach(TEST_FILE ${TEST_FILES})
	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
	add_executable(${TEST_NAME} ${TEST_FILE})
	target_link_libraries(${TEST_NAME} SparseANNLibrary OpenSSL::SSL OpenSSL::Crypto Parlay::parlay)
    target_compile_options(${TEST_NAME} PRIVATE -mavx2)
endforeach()
