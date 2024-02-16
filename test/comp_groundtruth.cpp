#include <iostream>
#include <fstream>
#include <cstdint>
#include <algorithm> // for min
#include <stdlib.h> // for strtoull
#include <cassert>

#include "forward_index.h"
#include "ground_truth.h"

int k = 100;

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [path to inserts .csr] [path to queries .csr] [path to outfile] [optional number of queries]" << std::endl;
		exit(0);
	}

	auto firstgt = ground_truth(argv[1], argv[2], k);
	std::cout << firstgt[0][0] << std::endl;

	size_t num_queries_to_read = (size_t)(-1);
	if (argc > 4) {
		num_queries_to_read = strtoull(argv[4], NULL, 10);
	}

	std::ofstream writer(argv[3]);
	if (!writer.is_open()) {
		std::cout << "Could not open outfile\n" << std::endl;
		exit(0);
	}

	forward_index<float> inserts(argv[1], "csr");
	if (inserts.points.size() == 0) {
		std::cout << "Could not read inserts file\n" << std::endl;
		exit(0);
	}

	forward_index<float> queries(argv[2], "csr", num_queries_to_read);
	if (queries.points.size() == 0) {
		std::cout << "Could not read queries file\n" << std::endl;
		exit(0);
	}
	
	uint32_t k = 10;
	auto gt = ground_truth(inserts, queries, k);

	// verifying the head of the ground truth is correct
	std::cout << "Verifying correctness..." << std::endl;
	int gt_wrong = -1;
	for (int i = 0; i < std::min((int)gt.size(), 20); i++) {
		int gt_count = 0;
		float gt_dist = forward_index<float>::dist(inserts.points[gt[i][0]], queries.points[i]);
		for (int j = 0; j < inserts.points.size(); j++) {
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
		std::cout << "Head:" << std::endl;
		for (int i = 0; i < k; i++) {
			float dist = forward_index<float>::dist(inserts.points[gt[gt_wrong][i]], queries.points[gt_wrong]);
			std::cout << "Point " << gt[gt_wrong][i] << " with distance " << dist << std::endl;
		}
	}

	// write ground truth to file
	uint32_t num_queries = queries.points.size();
	writer.write((char*)(&num_queries), sizeof(uint32_t));
	writer.write((char*)(&k), sizeof(uint32_t));
	for (int i = 0; i < gt.size(); i++) {
		writer.write((char*)(&gt[i][0]), k * sizeof(uint32_t));
	}
	writer.close();

	return 0;
}
