include_directories(${SparseANN_SOURCE_DIR}/include)

set(SOURCE_FILES
	linscan.cpp
	hashutil.cpp
)

add_library(SparseANNLibrary ${SOURCE_FILES})
