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

#include <immintrin.h>

#define TEST 4

int main(int argc, char **argv) {
	int k = 10;
	int overretrievals[10] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
	for (int i = 0; i < sizeof(overretrievals) / sizeof(overretrievals[0]); i++) {
		overretrievals[i] *= k;
	}

    srand(time(NULL));
    parlay::internal::timer timer;
    timer.start();

    std::cout << "Reading input files...\t" << std::flush;
	forward_index<float> inserts, queries;
    inserts = forward_index<float>("data/base_small.csr", "csr");
    queries = forward_index<float>("data/queries.dev.csr", "csr");
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    std::cout << "Computing ground truth of inputs...\t" << std::flush;
	parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gtd;
    gtd = ground_truth_with_distances(inserts, queries, k);
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    coord_order order("data/process/reorder/base_small_100000.ord", "ord");
    int comp_dims = 1400;
    timer.next_time();
    std::cout << "Applying reordering and compressing to " + std::to_string(comp_dims) + " dims...\t" << std::flush;
    inserts.reorder_dims(order.order_map);
    inserts = forward_index<float>::group_and_max(inserts, comp_dims);
    queries.reorder_dims(order.order_map);
    queries = forward_index<float>::group_and_max(queries, comp_dims);
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    // convert sparse vectors into nonzero bitvectors
    std::cout << "Constructing bitvector representations of vectors...\t" << std::flush;
    auto bitvectors = parlay::sequence<bitvector>::from_function(inserts.size(), [&] (size_t i) {
            auto bits = bitvector(DIV_ROUND_UP(inserts.size(), 256) * 256);
            bits.size = inserts.size();
            for (int j = 0; j < inserts.points[i].size(); j++) {
                bits.set(inserts.points[i][j].first);
            }
            return bits;
        });
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    // pick a few random queries
    // find the true mips distances between all points and the queries
    // find the approximate mips distances between all points and the queries
    // in particular, want to somehow evaluate the quality of the embedding in terms of its ability to preserve order of distances

    const size_t num_samples = 10;
    double exact_time = 0, approx_time = 0;
    for (int _num_samples = 0; _num_samples < num_samples; _num_samples++) {
        size_t sample_index = rand() % queries.size();
        auto& sample_query = queries.points[sample_index];
        auto query_bitvector = avx_bitvector::from_csr_vector(inserts.dims, queries.points[sample_index]);
        auto distances = std::make_pair<parlay::sequence<float>::uninitialized(inserts.size()), parlay::sequence<uint32_t>::uninitialized(inserts.size())>;
        
        timer.next_time();
        parlay::parallel_for(0, inserts.size(), [&] (size_t i) {
                distances.first[i] = inserts.dist(sample_query, inserts.points[i]);
            });
        exact_time += timer.next_time();

        parlay::parallel_for(0, inserts.size(), [&] (size_t i) {
                distances.second[i] = bitvectors[i].mips(query_bitvector);
            });
    }

	parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> tgtd;
	time_function("Computing ground truth of transformed inputs", [&] () {
		tgtd = ground_truth_with_distances(inserts, queries, overretrievals[(sizeof(overretrievals) / sizeof(overretrievals[0])) - 1]);
	});

	auto gt = parlay::sequence<parlay::sequence<uint32_t>>::from_function(gtd.size(), [&] (size_t i) {
		return parlay::sequence<uint32_t>::from_function(gtd[i].size(), [&] (size_t j) {
			return gtd[i][j].first;
		});
	});
	auto tgt = parlay::sequence<parlay::sequence<uint32_t>>::from_function(tgtd.size(), [&] (size_t i) {
		return parlay::sequence<uint32_t>::from_function(tgtd[i].size(), [&] (size_t j) {
			return tgtd[i][j].first;
		});
	});

	for (int overret : overretrievals) {
		double recall = get_recall(gt, tgt, k, overret);
		std::cout << "Recall " << overret << "@" << k << ":   \t" << recall << std::endl;
	}

	/*exit(0);

	double target_recall = 0.9;
	std::cout << "Target recall:\t" << target_recall << std::endl;
	uint64_t target_found = queries.size() * k * target_recall, total_found = 0;
	std::vector<int> num_not_found(queries.size(), k);*/
	
}
