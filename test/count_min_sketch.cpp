#include <cstdint>
#include <cstdlib>

#include "../include/forward_index.h"
#include "../include/count_min_sketch.h"

int main(int argc, char **argv) {
    if (argc <= 3) {
        std::cout << "Usage: " << argv[0] << " (# quant dims) (input file) (output file) [additional input/output file pairs]" << std::endl;
        exit(0);
    }

    uint32_t quant_dims = atoi(argv[1]);
    std::ifstream reader(argv[2]);
    if (!reader.is_open()) {
        std::cout << "Unable to read from " << argv[2] << std::endl;
        std::cout << "Aborting" << std::endl;
        exit(0);
    }
    uint64_t num_dims;
    reader.read((char*)(&num_dims), sizeof(uint64_t));
    reader.read((char*)(&num_dims), sizeof(uint64_t));

    std::cout << "Generating count min sketch from " << num_dims << " to " << quant_dims << " dimensions" << std::endl;
    count_min_sketch sketch(num_dims, quant_dims);

    for (int i = 2; i + 1 < argc; i += 2) {
        std::cout << std::endl << "Reading from " << argv[i] << std::endl;
        forward_index<float> vectors(argv[i], "csr");

        std::cout << "Loaded " << vectors.points.size() << " points of dimension " << vectors.dims << std::endl;
        if (vectors.dims != num_dims) {
            std::cout << "Error: found dimension " << vectors.dims << ", expected " << num_dims << std::endl;
            std::cout << "Skipping file" << std::endl;
            continue;
        }

        std::cout << "Transforming vectors to sketches" << std::endl;
        auto quant_vectors = parlay::sequence<parlay::sequence<float>>::from_function(
            vectors.points.size(),
            [&] (size_t i) {
                return sketch.transform_csr_to_qvec(vectors.points[i]);
            }
        );

        std::cout << "Saving to " << argv[i + 1] << std::endl;
        std::ofstream writer(argv[i + 1]);
        if (!writer.is_open()) {
            std::cout << "Unable to write to " << argv[i + 1] << std::endl;
            std::cout << "Skipping" << std::endl;
            continue;
        }

        uint32_t num_vecs = vectors.points.size();
        writer.write((char*)(&num_vecs), sizeof(uint32_t));
        writer.write((char*)(&quant_dims), sizeof(uint32_t));

        for (uint32_t i = 0; i < num_vecs; i++) {
            writer.write((char*)(&quant_vectors[i][0]), quant_dims * sizeof(float));
        }
        writer.close();
        std::cout << "Done" << std::endl;
    }

    return 0;
}
