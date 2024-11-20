#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>  // For file output
#include <algorithm> // For std::max
#include <cmath>     // For std::sqrt
#include <random>

#include "forward_index.h"

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <parlay/internal/get_time.h>
#include <openssl/rand.h>

/* TEST TYPES
0 - random sample from file
1 - generate uniform real dense vectors
2 - generate uniform real sparse vectors on uniform dimensions
*/
#define TEST_TYPE 0
#define TEST_TYPE_RANGE 3
#define MAX_POINTS 100
#define RANDOM_POINT_DIMS 20

// Function to compute the proportion of triangles that violate the inequality
// and record the violation amounts in a file for later analysis.
parlay::sequence<float> calculate_violations(const forward_index<float>& points) {
    size_t num_points = points.size();
    size_t num_points_sq = points.size() * points.size();
    size_t num_points_cu = points.size() * points.size() * points.size();

    auto violations = parlay::sequence<float>::uninitialized(num_points_cu);

    // Loop over all triples of points (x, y, z)
    parlay::parallel_for(0, num_points_cu, [num_points, num_points_sq, &points, &violations] (size_t i) {
        size_t x = i / num_points_sq;
        size_t y = (i / num_points) % num_points;
        size_t z = i % num_points;
        float violation = 0;
        if (x != y && y != z && x != z) {
            // Compute inner products
            float s_xy = forward_index<float>::dist(points[x], points[y]);
            float s_yz = forward_index<float>::dist(points[y], points[z]);
            float s_xz = forward_index<float>::dist(points[x], points[z]);

            // Compute the violation of the inequality: s(x, z) >= s(x, y) + s(y, z)
            violation = (s_xy + s_yz) - s_xz;
            if (violation < 0) violation = 0;
        }
        violations[x * num_points_sq + y * num_points + z] = violation;
    });

    return violations;
}

// Function to write the proportion and nonzero violations to a file
size_t write_violations_to_file(const std::string& output_file, const parlay::sequence<float>& violations) {
    auto nonzero_violations = parlay::filter(violations, [&] (float v) {
        return v != 0;
    });

    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return nonzero_violations.size();
    }

    // Write the proportion of violating triangles as the first line
    file << "Proportion of triangles with violations: " << (float)nonzero_violations.size() / violations.size() << "\n";

    // Write only the nonzero violations
    for (const double& violation : nonzero_violations) {
        if (violation > 0) {
            file << violation << "\n";
        }
    }

    file.close();

    return nonzero_violations.size();
}

int main(int argc, char *argv[]) {
    int test_type = TEST_TYPE;
    if (argc > 1) {
        test_type = atoi(argv[1]);
        if (test_type < 0 || test_type >= TEST_TYPE_RANGE) {
            std::cout << "Test type must be in range [0, " << TEST_TYPE_RANGE << ")" << std::endl;
            exit(0);
        }
    }

    forward_index<float> points;

    if (test_type == 0) {
        const char *input_file = "data/base_small.csr";
        auto all_points = forward_index<float>(input_file, "csr");
        points = forward_index<float>::sample(all_points, MAX_POINTS);
    }
    else if (test_type == 1) {
        parlay::random_generator gen;
        std::uniform_real_distribution<float> dis(0, 1);

        points.dims = RANDOM_POINT_DIMS;
        points.points = parlay::tabulate(MAX_POINTS, [&] (size_t i) {
            auto dense_vector = parlay::tabulate(RANDOM_POINT_DIMS, [&] (size_t j) {
                auto r = gen[i * RANDOM_POINT_DIMS + j];
                return std::make_pair<uint32_t, float>(j, dis(r));
            });
            return dense_vector;
        });
    }
    else if (test_type == 2) {
        parlay::random_generator gen;
        std::uniform_real_distribution<float> dis(0, 1);

        points.dims = RANDOM_POINT_DIMS * 10;
        std::uniform_int_distribution<size_t> int_dis(0, points.dims - 1);
        points.points = parlay::tabulate(MAX_POINTS, [&] (size_t i) {
            auto vals = parlay::sequence<float>(points.dims, 0);
            for (int j = 0; j < RANDOM_POINT_DIMS; j++) {
                auto r1 = gen[i * RANDOM_POINT_DIMS * 2 + j];
                auto r2 = gen[i * RANDOM_POINT_DIMS * 2 + RANDOM_POINT_DIMS + j];
                vals[int_dis(r1)] = dis(r2);
            }
            parlay::sequence<std::pair<uint32_t, float>> sparse_vector;
            sparse_vector.reserve(RANDOM_POINT_DIMS);
            for (int j = 0; j < points.dims; j++) {
                if (vals[j] != 0) {
                    std::pair<uint32_t, float> p = std::make_pair<uint32_t, float>(j, vals[j]);
                    sparse_vector.push_back(p);
                }
            }
            return sparse_vector;
        });
    }

    // Calculate the proportion of violating triangles and record violations
    auto violations = calculate_violations(points);

    // Output file to record the violation amounts
    std::string output_file;
    output_file = "violations.txt";

    // Write violations to the output file
    size_t num_violations = write_violations_to_file(output_file, violations);
    
    // Output the results
    std::cout << "Proportion of triangles that violate the inequality: " << (float)num_violations / violations.size() << std::endl;
    std::cout << "Violation amounts recorded in " << output_file << std::endl;

    return 0;
}
