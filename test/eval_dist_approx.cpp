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

#include "../include/count_min_sketch.h"
#include "../include/sinnamon_sketch.h"


parlay::sequence<int> rankify(const parlay::sequence<float>& vec) {
    auto n = vec.size();
    parlay::sequence<std::pair<float, int>> pair_vec(n);
    parlay::parallel_for(0, n, [&](size_t i) {
        pair_vec[i] = std::make_pair(vec[i], i);
    });

    parlay::sort_inplace(pair_vec);

    parlay::sequence<int> ranks(n);
    parlay::parallel_for(0, n, [&](size_t i) {
        ranks[pair_vec[i].second] = i + 1;
    });

    return ranks;
}

double spearman_rank_correlation(const parlay::sequence<float>& original, const parlay::sequence<float>& quantized) {
    if (original.size() != quantized.size()) {
        throw std::invalid_argument("Sequences must be of the same length");
    }

    auto n = original.size();
    auto rank_original = rankify(original);
    auto rank_quantized = rankify(quantized);

    auto d_sum = parlay::reduce(parlay::delayed_map(parlay::iota(n), [&](size_t i) {
        double d = static_cast<double>(rank_original[i]) - static_cast<double>(rank_quantized[i]);
        return d * d;
    }));

    double spearman_rho = 1 - (6 * d_sum) / (n * (n * n - 1));
    return spearman_rho;
}


struct test_params {
    int k;
    int *overretrievals;
    int num_overretrievals;
    int max_overretrieval;
    int num_evals;
    int eval_sample;
};

void evaluate_transformation(
    std::string transformation_name,
    forward_index<float>& transformed_inserts,
    forward_index<float>& transformed_queries,
    parlay::sequence<parlay::sequence<uint32_t>>& groundtruth,
    parlay::sequence<parlay::sequence<float>>& distances,
    test_params &params
) {
    parlay::internal::timer timer;
    timer.start();

    std::cout << "Computing ground truth of " << transformation_name << "...\t" << std::flush;
    auto transformed_groundtruth = ground_truth(transformed_inserts, transformed_queries, params.max_overretrieval);
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;
    for (int i = 0; i < params.num_overretrievals; i++) {
        double recall = get_recall(groundtruth, transformed_groundtruth, params.k, params.overretrievals[i]);
        std::cout << "Recall " << params.k << "@" << params.overretrievals[i] << ":\t" << recall << std::endl;
    }
    std::cout << std::endl;

    double avg_spearman_rank = 0, min_spearman_rank = 1, max_spearman_rank = -1;
    for (int i = 0; i < params.num_evals; i++) {
        std::cout << "\rComputing Spearman's rank correlation coefficients...\t" << i << "/" << params.num_evals << std::flush;

        auto transformed_distances = parlay::sequence<float>::from_function(params.eval_sample, [&] (size_t j) {
            return forward_index<float>::dist(transformed_queries.points[i], transformed_inserts.points[j]);
        });

        double spearman_rank = spearman_rank_correlation(distances[i], transformed_distances);
        avg_spearman_rank += spearman_rank;
        if (spearman_rank < min_spearman_rank) min_spearman_rank = spearman_rank;
        if (spearman_rank > max_spearman_rank) max_spearman_rank = spearman_rank;
    }
    avg_spearman_rank /= params.num_evals;
    std::cout << "\rComputing Spearman's rank correlation coefficients...\t" << params.num_evals << "/" << params.num_evals << " in " << timer.next_time() << " seconds" << std::endl;
    std::cout << "Min:\t" << min_spearman_rank << std::endl;
    std::cout << "Avg:\t" << avg_spearman_rank << std::endl;
    std::cout << "Max:\t" << max_spearman_rank << std::endl;
}


int main(int argc, char **argv) {
    test_params params;
    params.k = 10;
	int overretrievals[7] = {1, 2, 5, 10, 20, 50, 100};
    params.num_overretrievals = sizeof(overretrievals) / sizeof(overretrievals[0]);
	for (int i = 0; i < params.num_overretrievals; i++) {
		overretrievals[i] *= params.k;
	}
    params.overretrievals = overretrievals;
    params.max_overretrieval = overretrievals[params.num_overretrievals - 1];
    params.num_evals = 100;
    params.eval_sample = 1000;

    srand(time(NULL));
    parlay::internal::timer timer;
    timer.start();

    std::cout << "Reading input files...\t" << std::flush;
	forward_index<float> inserts, queries;
    inserts = forward_index<float>("data/base_small.csr", "csr");
    queries = forward_index<float>("data/queries.dev.csr", "csr", 1000);
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    std::cout << "Computing ground truth of inputs...\t" << std::flush;
    auto groundtruth = ground_truth(inserts, queries, params.k);
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    std::cout << "Computing sample distances...\t" << std::flush;
    auto distances = parlay::sequence<parlay::sequence<float>>::from_function(params.num_evals, [&] (size_t i) {
        return parlay::sequence<float>::from_function(params.eval_sample, [&] (size_t j) {
            return forward_index<float>::dist(queries.points[i], inserts.points[j]);
        });
    });
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;


    std::cout << std::endl << std::endl << "=== COUNT MIN SKETCH ===" << std::endl;
    timer.next_time();
    size_t count_min_quant_dims = 200;
    std::cout << "Generating count min sketch of dimension " << count_min_quant_dims << "...\t" << std::flush;
    auto count_min_sketch_200 = count_min_sketch(inserts.dims, count_min_quant_dims);
    auto count_min_inserts = forward_index<float>(count_min_quant_dims);
    auto count_min_queries = forward_index<float>(count_min_quant_dims);
    count_min_inserts.points = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>::from_function(inserts.size(), [&] (size_t i) {
        auto qvec = count_min_sketch_200.transform_csr_to_qvec(inserts.points[i]);
        return count_min_sketch_200.transform_qvec_to_qcsr(qvec);
    });
    count_min_queries.points = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>::from_function(queries.size(), [&] (size_t i) {
        auto qvec = count_min_sketch_200.transform_csr_to_qvec(queries.points[i]);
        return count_min_sketch_200.transform_qvec_to_qcsr(qvec);
    });
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    evaluate_transformation("count min sketch", count_min_inserts, count_min_queries, groundtruth, distances, params);


    std::cout << std::endl << std::endl << "=== SINNAMON SKETCH ===" << std::endl;
    timer.next_time();
    size_t sinnamon_quant_dims = 200;
    std::cout << "Generating sinnamon sketch of dimension " << sinnamon_quant_dims << "...\t" << std::flush;
    auto sinnamon_sketch_200 = sinnamon_sketch(inserts.dims, sinnamon_quant_dims);
    auto sinnamon_inserts = forward_index<float>(sinnamon_quant_dims);
    auto sinnamon_queries = forward_index<float>(sinnamon_quant_dims);
    sinnamon_inserts.points = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>::from_function(inserts.size(), [&] (size_t i) {
        auto qvec = sinnamon_sketch_200.transform_csr_to_qvec(inserts.points[i]);
        return sinnamon_sketch_200.transform_qvec_to_qcsr(qvec);
    });
    sinnamon_queries.points = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>::from_function(queries.size(), [&] (size_t i) {
        auto qvec = sinnamon_sketch_200.transform_csr_to_qvec(queries.points[i]);
        return sinnamon_sketch_200.transform_qvec_to_qcsr(qvec);
    });
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    evaluate_transformation("sinnamon sketch", sinnamon_inserts, sinnamon_queries, groundtruth, distances, params);


    std::cout << std::endl << std::endl << "=== 0/1 BITVECTOR ===" << std::endl;
    timer.next_time();
    std::cout << "Converting all nonzero coords to 1...\t" << std::flush;
    auto bitvector_inserts = forward_index<float>::copy(inserts);
    auto bitvector_queries = forward_index<float>::copy(queries);
    parlay::parallel_for(0, bitvector_inserts.size(), [&] (size_t i) {
        for (int j = 0; j < bitvector_inserts[i].size(); j++) {
            bitvector_inserts[i][j].second = 1;
        }
    });
    parlay::parallel_for(0, bitvector_queries.size(), [&] (size_t i) {
        for (int j = 0; j < bitvector_queries[i].size(); j++) {
            bitvector_queries[i][j].second = 1;
        }
    });
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    evaluate_transformation("0/1 bitvector", bitvector_inserts, bitvector_queries, groundtruth, distances, params);
}
