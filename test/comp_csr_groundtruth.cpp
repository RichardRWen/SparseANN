#include <iostream>
#include <fstream>
#include <cstdint>
#include <algorithm> // for min
#include <stdlib.h> // for strtoull
#include <cassert>
#include <limits>

#include "forward_index.h"
#include "ground_truth.h"

int k = 100;

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [path to inserts .csr] [path to queries .csr] [path to outfile] [optional number of queries]" << std::endl;
		exit(0);
	}

	/*auto firstgt = ground_truth(argv[1], argv[2], k);
	std::cout << firstgt[0][0] << std::endl;*/

	size_t num_queries_to_read = (size_t)(-1);
	if (argc > 4) {
		num_queries_to_read = strtoull(argv[4], NULL, 10);
	}

	std::ofstream writer(argv[3]);
	if (!writer.is_open()) {
		std::cout << "Could not open outfile\n" << std::endl;
		exit(0);
	}

	std::cout << "Reading inserts..." << std::endl;
	forward_index<float> inserts(argv[1], "csr");
	if (inserts.points.size() == 0) {
		std::cout << "Could not read inserts file\n" << std::endl;
		exit(0);
	}
    std::cout << "Detected " << inserts.size() << " points of dimension " << inserts.dims << std::endl;

	std::cout << "Reading queries..." << std::endl;
	forward_index<float> queries(argv[2], "csr", num_queries_to_read);
	if (queries.points.size() == 0) {
		std::cout << "Could not read queries file\n" << std::endl;
		exit(0);
	}
    std::cout << "Detected " << queries.size() << " points of dimension " << queries.dims << std::endl;
	
	std::cout << "Computing gt..." << std::endl;
	uint32_t k = 10;
	auto gt = ground_truth_with_distances(inserts, queries, k);

	// verifying the head of the ground truth is correct
	/*std::cout << "Verifying correctness..." << std::endl;
	int gt_wrong = -1;
    int num_missing_neighbors = 0;
    int num_nearest_neighbors;
	for (int i = 0; i < std::min((int)gt.size(), 20); i++) {
		int gt_count = 0;
		float gt_dist = 0;
        num_nearest_neighbors = 0;
        for (; num_nearest_neighbors < k; num_nearest_neighbors++) {
            if (gt[i][num_nearest_neighbors].first < inserts.size()) gt_dist = gt[i][num_nearest_neighbors].second;
            else break;
        }
        num_missing_neighbors += k - num_nearest_neighbors;

		for (int j = 0; j < num_nearest_neighbors; j++) {
			if (forward_index<float>::dist(inserts.points[j], queries.points[i]) < gt_dist) gt_count++;
		}

		if (gt_count >= k) {
			gt_wrong = i;
			break;
		}
	}
	if (gt_wrong < 0) {
		std::cout << "Head is correct" << std::endl;
	}
	else {
		std::cout << "Head is incorrect at query " << gt_wrong << std::endl;
        std::cout << "GT-nearest neighbors: " << num_nearest_neighbors << std::endl;
        std::cout << "Top-k distance is " << (num_nearest_neighbors == 0 ? 0 : gt[gt_wrong][num_nearest_neighbors - 1].second) << std::endl;
		std::cout << "Head:" << std::endl;
		for (int i = 0; i < k; i++) {
			std::cout << "Point " << gt[gt_wrong][i].first << " with distance " << gt[gt_wrong][i].second << std::endl;
		}
	}
    std::cout << "Ground truth has " << num_missing_neighbors << " fewer neighbors than desired" << std::endl;
    std::cout << "Desired nearest neighbors: " << k * gt.size() << std::endl;*/

	std::cout << "writing to file " << argv[3] << std::endl;
	// write ground truth to file
	uint32_t num_queries = queries.points.size();
	writer.write((char*)(&num_queries), sizeof(uint32_t));
	writer.write((char*)(&k), sizeof(uint32_t));
	for (int i = 0; i < gt.size(); i++) {
        for (int j = 0; j < k; j++) {
            writer.write((char*)(&gt[i][j].first), sizeof(uint32_t));
        }
	}
	writer.close();

	return 0;
}
