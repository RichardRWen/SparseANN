#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>

#include "synthetic.h"

int main(int argc, const char **argv) {
    if (argc < 8) {
        std::cout << "Usage: " << argv[0] << " [num points (1000000)] [num queries (10000)] [dimensions (128)] [clusters (10)] [clustering factor (0, 0.5)] [base outfile (base.fbin)] [query outfile (queries.fbin)]" << std::endl;
        return 0;
    }

    int num_points = atoi(argv[1]);  // Total number of points
    int num_queries = atoi(argv[2]); // Total number of queries
    int dimensions = atoi(argv[3]);  // Points are in 3D space
    int initial_points = atoi(argv[4]);  // Start with 5 initial points
    float clustering_factor = std::stof(argv[5]);  // Control how strongly new points prefer to connect to existing clusters
    const char *base_filename = argv[6];
    const char *query_filename = argv[7];

    std::vector<std::vector<float>> point_set = generate_points_barabasi_albert(num_points + num_queries, dimensions, initial_points, clustering_factor);
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(point_set), std::end(point_set), rng);

    std::ofstream base_writer(base_filename);
    std::ofstream query_writer(query_filename);
    if (!base_writer.is_open() || !query_writer.is_open()) {
        std::cout << "Unable to open files for writing" << std::endl;
        return 0;
    }

    uint32_t num_dims = dimensions, num_vecs = num_points;
    base_writer.write((char*)(&num_vecs), sizeof(uint32_t));
    base_writer.write((char*)(&num_dims), sizeof(uint32_t));

    for (size_t i = 0; i < num_points; i++) {
        base_writer.write((char*)(&point_set[i][0]), num_dims * sizeof(float));
    }

    base_writer.close();

    num_vecs = num_queries;
    query_writer.write((char*)(&num_vecs), sizeof(uint32_t));
    query_writer.write((char*)(&num_dims), sizeof(uint32_t));

    for (size_t i = num_points; i < point_set.size(); i++) {
        query_writer.write((char*)(&point_set[i][0]), num_dims * sizeof(float));
    }

    query_writer.close();

    return 0;
}
