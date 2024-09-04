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

#include "../include/vector_utils.h"


int main(int argc, char **argv) {
    int k = 10;

    srand(time(NULL));
    parlay::internal::timer timer;
    timer.start();

    std::cout << "Reading input files...\t" << std::flush;
	forward_index<float> inserts, queries;
    inserts = forward_index<float>("data/base_small.csr", "csr", 100000);
    //queries = forward_index<float>::copy(inserts);
    queries = forward_index<float>("data/queries.dev.csr", "csr");
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    std::cout << "Computing ground truth of inputs...\t" << std::flush;
    auto groundtruth = ground_truth_with_distances(inserts, queries, k);
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    std::unordered_map<uint32_t, uint32_t> top_1;
    std::unordered_map<uint32_t, uint32_t> top_k;
    for (int i = 0; i < groundtruth.size(); i++) {
        top_1[groundtruth[i][0].first]++;
        for (int j = 0; j < groundtruth[i].size(); j++) {
            top_k[groundtruth[i][j].first]++;
        }
    }
    std::vector<std::pair<uint32_t, uint32_t>> top_1_freq(top_1.begin(), top_1.end());
    std::sort(top_1_freq.begin(), top_1_freq.end(),
        [] (const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
            return b.second < a.second;
        });
    std::vector<std::pair<uint32_t, uint32_t>> top_k_freq(top_k.begin(), top_k.end());
    std::sort(top_k_freq.begin(), top_k_freq.end(),
        [] (const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
            return b.second < a.second;
        });

    std::cout << queries.size() << " total queries" << std::endl;
    std::cout << "Top 1 appearances: " << std::endl;
    for (int i = 0; i < std::min((int)top_1_freq.size(), 10); i++) {
        std::cout << i << ":\t" << top_1_freq[i].first << " with " << top_1_freq[i].second << " appearances" << std::endl;
    }
    std::cout << "Top " << k << " appearances: " << std::endl;
    for (int i = 0; i < std::min((int)top_k_freq.size(), 10); i++) {
        std::cout << i << ":\t" << top_k_freq[i].first << " with " << top_k_freq[i].second << " appearances" << std::endl;
    }
}
