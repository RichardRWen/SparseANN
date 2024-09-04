#ifndef _JL_TRANSFORM_H_
#define _JL_TRANSFORM_H_

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <parlay/sequence.h>

// Function to generate a random Gaussian matrix
std::vector<std::vector<float>> generate_random_projection_matrix(int original_dim, int reduced_dim) {
    std::vector<std::vector<float>> projection_matrix(reduced_dim, std::vector<float>(original_dim));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f); // Gaussian distribution

    for (int i = 0; i < reduced_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            projection_matrix[i][j] = dist(gen);
        }
    }

    return projection_matrix;
}

// Function to apply JL Transform on a vector
parlay::sequence<float> apply_jl_transform(const parlay::sequence<float>& vec, const std::vector<std::vector<float>>& projection_matrix) {
    int reduced_dim = projection_matrix.size();
    int original_dim = vec.size();
    parlay::sequence<float> reduced_vec(reduced_dim, 0.0f);

    for (int i = 0; i < reduced_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            reduced_vec[i] += projection_matrix[i][j] * vec[j];
        }
    }

    return reduced_vec;
}

#endif
