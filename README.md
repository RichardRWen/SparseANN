To build,
```
mkdir build
cd build
cmake ..
cmake .. -B build-debug -DCMAKE_BUILD_TYPE=Debug
```

To run from the build folder,
```
make clean && make
./test/unit_linscan
```
