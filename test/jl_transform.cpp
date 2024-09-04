#include <iostream>
#include <cstring>

#include "forward_index.h"

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <parlay/internal/get_time.h>
#include <openssl/rand.h>

#include "../include/vector_utils.h"
#include "../include/jl_transform.h"


int main(int argc, char **argv) {
    parlay::internal::timer timer;
    timer.start();

    std::cout << "Reading input files...\t" << std::flush;
	forward_index<float> inserts, queries;
    inserts = forward_index<float>("data/base_small.csr", "csr");
    queries = forward_index<float>("data/queries.dev.csr", "csr");
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    std::cout << std::endl << std::endl << "=== JL TRANSFORM ===" << std::endl;
    timer.next_time();
    size_t jl_transform_dims = 1000;
    std::cout << "Generating JL transform of dimension " << jl_transform_dims << "...\t" << std::flush;
    auto projection_matrix = generate_random_projection_matrix(inserts.dims, jl_transform_dims);
    forward_index<float> jl_inserts, jl_queries;
    jl_inserts.dims = jl_queries.dims = jl_transform_dims;
    jl_inserts.points = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>::from_function(inserts.size(), [&] (size_t i) {
        auto insert_vec = csr_to_vec(inserts[i], inserts.dims);
        auto jl_transform = apply_jl_transform(insert_vec, projection_matrix);
        return vec_to_csr(jl_transform);
    });
    jl_queries.points = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>::from_function(queries.size(), [&] (size_t i) {
        auto query_vec = csr_to_vec(queries[i], inserts.dims);
        auto jl_transform = apply_jl_transform(query_vec, projection_matrix);
        return vec_to_csr(jl_transform);
    });
    std::cout << "Done in " << timer.next_time() << " seconds" << std::endl;

    jl_inserts.write_to_file("data/jl_1000_base_small.fbin", "vec");
    jl_queries.write_to_file("data/jl_1000_queries.fbin", "vec");
}
