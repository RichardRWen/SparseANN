include_directories(${SparseANN_SOURCE_DIR}/include)

set(SOURCE_FILES
	hashutil.cpp
	#ground_truth.cpp
)

add_library(SparseANNLibrary ${SOURCE_FILES})
