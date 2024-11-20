#ifndef __SYNTHETIC_H__
#define __SYNTHETIC_H__

#include <random>
#include <vector>
#include <cmath>
#include <iostream>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

using point_t = std::vector<float>;

template <typename Rng, typename Distr>
point_t generate_random_point(int dimensions, Rng &rng, Distr &dist) {
    point_t point;
    point.resize(dimensions);
    for (int i = 0; i < dimensions; i++) {
        point[i] = dist(rng);
    }
    return point;
}

std::vector<point_t> generate_points_barabasi_albert(int num_points, int dimensions, int initial_points, float clustering_factor) {
    if (clustering_factor > 0.5) clustering_factor = 0.5;
    if (clustering_factor < 0) clustering_factor = 0;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uniform(0, 1);
    std::uniform_real_distribution<float> init_dist(clustering_factor, 1 - clustering_factor);
    std::uniform_real_distribution<float> dist(-clustering_factor, clustering_factor);

    std::vector<point_t> points;

    for (int i = 0; i < initial_points; ++i) {
        points.push_back(generate_random_point(dimensions, rng, init_dist));
    }

    std::vector<int> degrees(initial_points, 1);

    for (int i = initial_points; i < num_points; ++i) {
        point_t new_point = generate_random_point(dimensions, rng, dist);
        float r = uniform(rng) * i;
        for (int k = 0; k < initial_points; ++k) {
            r -= degrees[k];
            if (r <= 0) {
                for (int l = 0; l < dimensions; ++l) {
                    new_point[l] += points[k][l];
                }
                degrees[k]++;
                break;
            }
        }

        points.push_back(new_point);
    }

    return points;
}

std::vector<point_t> generate_points_gaussian_mixture(int num_points, int dimensions, int clusters, float std_dev) {
    if (std_dev > 0.5) std_dev = 0.5;
    if (std_dev < 0) std_dev = 0;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uniform(0, 1);
    std::uniform_real_distribution<float> init_dist(0, 1);
    std::normal_distribution<float> dist(0, std_dev);

    std::vector<point_t> points;

    for (int i = 0; i < clusters; ++i) {
        points.push_back(generate_random_point(dimensions, rng, init_dist));
    }

    std::vector<int> degrees(clusters, 1);

    for (int i = clusters; i < num_points; ++i) {
        point_t new_point = generate_random_point(dimensions, rng, dist);
        float r = uniform(rng) * i;
        for (int k = 0; k < clusters; ++k) {
            r -= degrees[k];
            if (r <= 0) {
                for (int l = 0; l < dimensions; ++l) {
                    new_point[l] += points[k][l];
                }
                degrees[k]++;
                break;
            }
        }

        points.push_back(new_point);
    }

    return points;
}

std::vector<point_t> generate_points_heatmap(int num_points, int dimensions, int clusters, float std_dev) {
    if (std_dev > 0.5) std_dev = 0.5;
    if (std_dev < 0) std_dev = 0;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uniform(0, 1);
    std::uniform_real_distribution<float> init_dist(0, 1);
    std::normal_distribution<float> dist(0, std_dev);

    std::vector<point_t> points;

    for (int i = 0; i < clusters; ++i) {
        points.push_back(generate_random_point(dimensions, rng, init_dist));
    }

    for (int i = clusters; i < num_points; ++i) {
        point_t new_point = generate_random_point(dimensions, rng, dist);
        float r = uniform(rng) * i;
        size_t k = r;
        for (int l = 0; l < dimensions; ++l) {
            new_point[l] += points[k][l];
        }

        points.push_back(new_point);
    }

    return points;
}

parlay::sequence<point_t> generate_points_knn(size_t num_points, size_t dimensions, size_t k, double universe_scale = 10) {
    if (universe_scale < 1) universe_scale = 1;
    size_t universe_size = num_points * universe_scale;
    parlay::random_generator rng;
    std::uniform_real_distribution<float> uniform(0, 1);
    auto universe = parlay::tabulate<point_t>(universe_size, [&] (size_t i) {
        auto thread_rng = rng[i];
        return generate_random_point(dimensions, thread_rng, uniform);
    });

    auto knn_points = parlay::tabulate<point_t>(num_points, [&] (size_t i) {
        return universe[i];
    });

    return knn_points;
}

#endif
