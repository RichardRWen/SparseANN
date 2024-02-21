#include <iostream>

#include "forward_index.h"
#include "ground_truth.h"

using id_type = uint32_t;
using gt_type = parlay::sequence<parlay::sequence<id_type>>;

// solely a sanity check test - we are just testing the gt with itself
double get_recall_linscan(gt_type& gt, char *inserts_file, char *queries_file, int k = 10) {
	std::cout << "Test: linscan" << std::endl;

	forward_index<float> inserts(inserts_file, "csr");
	forward_index<float> queries(queries_file, "csr");

	auto neighbors = ground_truth(inserts, queries, k);

	double recall = get_recall(gt, neighbors, k);
	
	std::cout << "Recall: " << recall << std::endl;
	return recall;
}

void compress_vectors(forward_index<float>& index, uint32_t comp_factor) {
	uint32_t comp_dims = index.dims / comp_factor;
	float *buffer = new float[comp_dims];
	for (int i = 0; i < index.points.size(); i++) {
		bzero(buffer, index.dims * sizeof(float));
		for (auto coord : index.points[i]) {
			if (buffer[coord.first / comp_factor] < coord.second) buffer[coord.first / comp_factor] = coord.second;
		}

		parlay::sequence<std::pair<id_type, float>> comp_point;
		for (id_type j = 0; j < comp_factor; j++) {
			if (buffer[j] != 0) comp_point.push_back(std::make_pair(j, buffer[j]));
		}
		index.points[i] = comp_point;
	}
}

double get_recall_compressed(gt_type& gt, char *inserts_file, char *queries_file, int k = 10) {
	std::cout << "Test: compressed" << std::endl;

	forward_index<float> inserts(inserts_file, "csr");
	forward_index<float> queries(queries_file, "csr");

	uint32_t comp_factor = 150;
	compress_vectors(inserts, comp_factor);
	compress_vectors(queries, comp_factor);

	auto neighbors = ground_truth(inserts, queries, k);

	double recall = get_recall(gt, neighbors, k);
	
	std::cout << "Recall: " << recall << std::endl;
	return recall;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " [inserts csr file] [queries csr file]" << std::endl;
	}
	
	int k = 10;
	forward_index<float> inserts(argv[1], "csr");
	forward_index<float> queries(argv[2], "csr");
	gt_type gt = ground_truth(inserts, queries, k);

	//get_recall_linscan(gt, argv[1], argv[2], 10);
	get_recall_compressed(gt, argv[1], argv[2], 10);
}
