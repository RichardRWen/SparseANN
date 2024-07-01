#include <iostream>
#include <cstring>
#include <utility>
#include <algorithm>

#include <unordered_set>

#include "forward_index.h"
#include "ground_truth.h"
#include "coord_order.h"
#include "bitvector.h"
#include "test_util.h"

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

int main(int argc, char **argv) {
    const int k = 10;
    const int over_k = 2 * k;

	forward_index<float> inserts, queries;
	time_function("Reading input files", [&] () {
		inserts = forward_index<float>("data/base_small.csr", "csr");
		queries = forward_index<float>("data/queries.dev.csr", "csr");
	});

	parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gtd;
	time_function("Querying normal inverted index", [&] () {
		gtd = ground_truth_with_distances(inserts, queries, k);
	});

    coord_order order("data/processed/reorder/base_small_100000.ord", "ord");
    int comp_dims = 120;//1400;
    time_function("Applying reordering", [&] () {
        inserts.reorder_dims(order.order_map);
        queries.reorder_dims(order.order_map);
    });

    std::vector<std::vector<uint32_t>> posting_lists(inserts.dims);
	time_function("Constructing weightless posting lists", [&] () {
		for (int i = 0; i < inserts.size(); i++) {
            for (int j = 0; j < inserts.points[i].size(); j++) {
                posting_lists[inserts.points[i][j].first].push_back(i);
            }
        }
	});

    parlay::sequence<avx_bitvector> insert_bitvectors;
    time_function("Constructing insert bitvectors", [&] () {
        /*insert_bitvectors = parlay::sequence<avx_bitvector>::from_function(inserts.size(), [&] (size_t i) {
            return avx_bitvector::from_sparse_coords(inserts.points[i], inserts.dims);
        });*/
        for (int i = 0; i < inserts.size(); i++) {
            std::cout << "\r" << i << std::flush;
            insert_bitvectors.push_back(avx_bitvector::from_sparse_coords(inserts.points[i], inserts.dims));
        }
    });
    parlay::sequence<avx_bitvector> query_bitvectors;
    time_function("Constructing query bitvectors", [&] () {
        query_bitvectors = parlay::sequence<avx_bitvector>::from_function(queries.size(), [&] (size_t i) {
            return avx_bitvector::from_sparse_coords(queries.points[i], queries.dims);
        });
    });

    auto get_neighbors = [&] (uint32_t query_index) {
        auto &query = queries.points[query_index];
        auto &query_bitvector = query_bitvectors[query_index];

        std::unordered_set<uint32_t> candidates;
        for (int i = 0; i < query.size(); i++) {
            candidates.insert(posting_lists[query[i].first].begin(), posting_lists[query[i].first].end());
        }
        
        parlay::sequence<std::pair<uint32_t, uint32_t>> heap;
        for (uint32_t candidate : candidates) {
            //uint32_t dot_prod = query_bitvector.dot_product(insert_bitvectors[candidate]);
            uint32_t dot_prod = avx_bitvector::dot_product(query_bitvector, insert_bitvectors[candidate]);
            if (heap.size() == over_k && dot_prod <= heap[0].first) continue;

            heap.push_back(std::make_pair(candidate, dot_prod));
            std::push_heap(heap.begin(), heap.end(), [] (auto &a, auto &b) {
                return a.second > b.second;
            });
            std::pop_heap(heap.begin(), heap.end(), [] (auto &a, auto &b) {
                return a.second > b.second;
            });
            heap.pop_back();
        }

        //parlay::sort_inplace(heap, [] (

        return heap;
    };

    parlay::sequence<parlay::sequence<std::pair<uint32_t, uint32_t>>> query_answers;
	time_function("Querying bitvector inverted index", [&] () {
        query_answers = parlay::sequence<parlay::sequence<std::pair<uint32_t, uint32_t>>>::from_function(
            queries.size(), [&] (size_t i) {
                return get_neighbors(i);
            }
        );
	});

	/*for (int overret : overretrievals) {
		double recall = get_recall(gt, tgt, k, overret);
		std::cout << "Recall " << overret << "@" << k << ":   \t" << recall << std::endl;
	}*/

	/*exit(0);

	double target_recall = 0.9;
	std::cout << "Target recall:\t" << target_recall << std::endl;
	uint64_t target_found = queries.size() * k * target_recall, total_found = 0;
	std::vector<int> num_not_found(queries.size(), k);*/
	
}
