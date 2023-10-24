To build
```
mkdir build
cd build
cmake ..
cmake .. -B build-debug -DCMAKE_BUILD_TYPE=Debug
```

To run from either build/ or build/build-debug/
```
make clean && make
./test/unit_linscan
```

For base/query files, you can look on fern in /ssd1/anndata/bigann/data/sparse/
For convenience, you can make symlinks in build/, build-debug/, and SparseParlayANN/algorithms/vamana/
```
ln -s /ssd1/anndata/bigann/data/sparse data
```

To convert a .csr base file to a .fvecs file of compressed vectors via Sinnamon
Usage: ./max_sketch [path to .csr] [path to outfile] [compressed dims] [number of hash functions] [optional seed]
In the following examples, we use seed 0. This can be any seed you want, but you'll want to make sure you use the same seed for compressing the base file and query file. You'll also want to use the same compressed dims and number of hash functions.
```
./test/max_sketch data/base_small.csr data/processed/base_small_200_1.fvecs 200 1 0
./test/max_sketch data/base_1M.csr data/processed/base_1M_200_1.fvecs 200 1 0
./test/max_sketch data/queries.dev.csr data/processed/queries_200_1.fvecs 200 1 0
```

To profile how good the compressed vectors are
```
./test/profile_sketch data/base_small.csr data/base_small_200_1.fvecs
./test/profile_sketch data/queries.dev.csr data/queries_200_1.fvecs
```

To use compressed vectors to build a vamana graph
```
cd SparseParlayANN/algorithms/vamana
make clean && make
./neighbors -R 256 -L 64 -a 1.2 -graph_outfile data/graph/base_small_200_1 -data_type float -file_type vec -dist_func mips -base_path data/processed/base_small_200_1.fvecs
./neighbors -R 256 -L 64 -a 1.2 -graph_path data/graph/base_small_200_1 -query_path data/processed/queries_200_1.fvecs -gt_path data/base_small.dev.gt -data_type float -file_type vec -dist_func mips -base_path data/processed/base_small_200_1.fvecs
./neighbors -R 256 -L 64 -a 1.2 -graph_path data/graph/base_small_200_1 -query_path data/queries.dev.csr -gt_path data/base_small.dev.gt -data_type float -file_type vec -dist_func sparse-mips -base_path data/base_small.csr
```
The first run of ./neighbors builds a graph using the compressed vectors.
The second run queries using compressed vector MIPS.
The third run queries using exact vector MIPS.
