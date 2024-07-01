#include <iostream>
#include <algorithm> // lower_bound

#include "forward_index.h"
#include "ground_truth.h"

using id_type = uint32_t;
using gt_type = parlay::sequence<parlay::sequence<id_type>>;
using gtd_type = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>;

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

double get_ball(gtd_type& gtd, double ball_size, int k, int overret) {
	double point_count = 0;
	for (int i = 0; i < gtd.size(); i++) {
		auto ball_radius = std::make_pair<uint32_t, float>(0, gtd[i][k - 1].second * ball_size);
		size_t j = 0;
		while (gtd[i][j].second >= ball_radius.second) j++;
		/*size_t j = std::upper_bound(gtd[i].begin(), gtd[i].end(), ball_radius, [] (std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) -> bool {
			return a.second > b.second;
		}) - gtd[i].begin();*/

		point_count += j;
	}
	return point_count / gtd.size();
}

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " [inserts csr file] [queries csr file]" << std::endl;
	}
	
	int k = 10;
	forward_index<float> inserts(argv[1], "csr");
	forward_index<float> queries(argv[2], "csr");
	//gt_type gt = ground_truth(inserts, queries, k);
	gtd_type gtd = ground_truth_with_distances(inserts, queries, 100000);
	float ball_sizes[6] = {0.1, 0.25, 0.5, 1, 2, 4};
	double point_counts[6];
	parlay::parallel_for(0, 6, [&] (size_t i) {
		double ball_size = 1.0 / (ball_sizes[i] + 1);
		point_counts[i] = get_ball(gtd, ball_size, k, 100000);
	}, 1);
	for(int i = 0; i < 6; i++) {
		double ball_size = 1.0 / (ball_sizes[i] + 1);
		std::cout << "for E = " << ball_sizes[i] << ": " << point_counts[i] << std::endl;
	}


	//get_recall_linscan(gt, argv[1], argv[2], 10);
	//get_recall_compressed(gt, argv[1], argv[2], 10);
}
