#include <iostream>
#include <cstring>

#include "forward_index.h"
#include "ground_truth.h"
#include "coord_order.h"
#include "test_util.h"
#include "bitvector.h"

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <parlay/internal/get_time.h>
#include <openssl/rand.h>


int main(int argc, char **argv) {
    srand(time(NULL));
    parlay::internal::timer timer;
    timer.start();

    std::cout << "Reading input files...\t" << std::flush;
	forward_index<float> inserts;
    inserts = forward_index<float>("data/base_small.csr", "csr");
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    int violations = 0;
    int total_triplets = 0;

    int sample_size = 1000;
    parlay::parallel_for(0, sample_size, [&] (size_t i) {
        for (size_t j = i + 1; j < sample_size; ++j) {
            for (size_t k = j + 1; k < sample_size; ++k) {
                double d_ab = forward_index<float>::dist(inserts.points[i], inserts.points[j]);
                double d_bc = forward_index<float>::dist(inserts.points[j], inserts.points[k]);
                double d_ac = forward_index<float>::dist(inserts.points[i], inserts.points[k]);

                if (d_ac > d_ab + d_bc) {
                    violations++;
                }
                total_triplets++;
            }
        }
    });

    std::cout << "Total Violations: " << violations << " out of " << total_triplets << " triplets" << std::endl;
    return violations == 0;
}
