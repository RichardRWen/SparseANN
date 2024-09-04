#ifndef _RABITQ_H_
#define _RABITQ_H_

#include <iostream>
#include <vector>
#include <random>

#include <parlay/sequence.h>

// Function to generate random vectors
std::vector<float> generate_random_vector(int dimension) {
    std::vector<float> vec(dimension);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < dimension; ++i) {
        vec[i] = dis(gen);
    }

    return vec;
}

// Function to quantize a vector using RaBitQ, returning a full vector of floats
parlay::sequence<float> rabitq_quantize(const parlay::sequence<float>& vec) {
    parlay::sequence<float> quantized_vec(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        quantized_vec[i] = vec[i] > 0 ? 1.0f : 0.0f;
    }
    return quantized_vec;
}

#endif
