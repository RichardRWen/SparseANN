#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "forward_index.h"
#include "ground_truth.h"
#include "coord_order.h"
#include "test_util.h"

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

int main(int argc, char **argv) {
	forward_index<float> inserts, queries;
	time_function("Reading input files", [&] () {
		//inserts = forward_index<float>("data/base_1M.csr", "csr");
		//inserts = forward_index<float>("/ssd1/trubel/DBLP/dblp_v14_TRAIN.csr", "csr");
		inserts = forward_index<float>("/ssd1/trubel/rnaseqdata/1M_neurons_filtered_gene_bc_matrices_h5_TRAIN.csr", "csr", 10000);
		queries = forward_index<float>(inserts.dims);
	});

	const int individual_histograms = 5;
	const int avged_histograms = 20;
	int indexes[individual_histograms];

	srand(time(NULL));
	for (int i = 0; i < 25; i++) {
		if (i < individual_histograms) {
			indexes[i] = rand() % inserts.points.size();
			queries.points.push_back(inserts.points[indexes[i]]);
		}
		else queries.points.push_back(inserts.points[rand() % inserts.points.size()]);
	}

	parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gtd;
	time_function("Computing ground truth of inputs", [&] () {
		gtd = ground_truth_with_distances(inserts, queries, inserts.points.size());
	});

	const float interval = 0.02;
    
	for (int i = 0; i < individual_histograms; i++) {
		std::ofstream writer("logs/hist_" + std::to_string(indexes[i]) + ".csv");
		writer << "interval,frequency\n" << std::flush;
		int j, k = 0;
		for (float inc = 1 - interval; inc >= 0; inc -= interval) {
			j = k;
			while (gtd[i][k].second && k < gtd[i].size() && gtd[i][k].second >= inc * gtd[i][0].second) k++;
			writer << inc + interval << "," << (float)(k - j) / gtd[i].size() << std::endl;
		}
        writer << "0," << (float)(gtd[i].size() - k) / gtd[i].size() << std::endl;
		writer.close();
	}

	const int num_intervals = (int)(1 / interval);
	int total_counts[num_intervals];
	int total_points = 0;
	for (int i = individual_histograms; i < individual_histograms + avged_histograms; i++) {
		int j, k = 0;
		for (int l = 0; l < num_intervals; l++) {
			float inc = (1 - (l + 1) * interval);
			j = k;
			while (k < gtd[i].size() && gtd[i][k].second >= inc * gtd[i][0].second) k++;
			total_counts[l] += (k - j);
		}
		total_points += gtd[i].size();
	}
	std::ofstream writer("logs/hist_avg.csv");
	writer << "interval,frequency\n";
	for (int i = 0; i < num_intervals; i++) {
		writer << (1 - (i + 1) * interval) << "," << (float)total_counts[i] / total_points << "\n";
	}
	writer.close();
}
