#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ctype.h>
#include <sys/time.h>

#include <cassert>

#include <openssl/rand.h>

#include "../include/linscan.h"

#define NUM_VERBOSE_INTERVALS 100ull

int main(int argc, char **argv) {
	int c, wflag = 0, vflag = 0;
	int vector_len = 5, num_inserts = 20, num_queries = 20;
	/* You should be able to tweak
	 * The length of a vector (aka the number of keywords in the dictionary)
	 * The number of document vectors to insert into the database
	 * The number of queries to make
	 * (The distribution of keywords in a document vector)
	 * (The distribution of keywords in a query vector)
	 * (If I want to eventually put this all in one test file, maybe the type of data structure to test)
	 */
	while ((c = getopt(argc, argv, "wvl:i:q:")) != -1) {
		switch (c) {
		case 'w': // complete work shown
			wflag = vflag = 1;
			break;
		case 'v': // verbose
			vflag = 1;
			break;
		case 'l': // length of a vector
			vector_len = atoi(optarg);
			if (vector_len <= 0) {
				fprintf(stderr, "Vector length must be positive. (Given: \"%d\")\n", vector_len);
				abort();
			}
			break;
		case 'i': // number of inserts to make
			num_inserts = atoi(optarg);
			if (num_inserts < 0) {
				fprintf(stderr, "Number of inserts must be nonnegative. (Given: \"%d\")\n", num_inserts);
				abort();
			}
			break;
		case 'q': // number of queries to make
			num_queries = atoi(optarg);
			if (num_queries < 0) {
				fprintf(stderr, "Number of queries must be nonnegative. (Given: \"%d\")\n", num_queries);
				abort();
			}
			break;
		case '?':
			switch (optopt) {
			case 'l':
			case 'i':
			case 'q':
				fprintf(stderr, "Option -%c requires an argument.\n", optopt);
				abort();
			default:
				if (isprint(optopt)) fprintf(stderr, "Unknown option \"-%c\".\n", optopt);
				else fprintf(stderr, "Unknown option character \\x%x.\n", optopt);
				abort();
			}
		}
	}
	for (int i = optind; i < argc; i++) {
		fprintf(stderr, "Non-option argument %s\n", argv[i]);
	}
	if (optind < argc) {
		fprintf(stderr, "For usage details, type \"%s --help\"\n", argv[0]);
		abort();
	}

	uint64_t verbose_point, verbose_interval;
	struct timeval start_time, end_time, start_interval_time, end_interval_time;

	printf("--Vector length:  %d\n", vector_len);
	printf("--Num of inserts: %d\n", num_inserts);
	printf("--Num of queries: %d\n", num_queries);

	fprintf(stderr, "Initializing data structures... ");
	inverted_index inv_index;
	inverted_index_init(&inv_index, vector_len);
	fprintf(stderr, "Done\n");

	fprintf(stderr, "Generating insert set... ");
	verbose_interval = 1;
	verbose_point = num_inserts * verbose_interval / NUM_VERBOSE_INTERVALS - 1;
	gettimeofday(&start_interval_time, NULL);
	int **insert_vectors = new int*[num_inserts];
	for (int i = 0; i < num_inserts; i++) {
		insert_vectors[i] = new int[vector_len];
		RAND_bytes((unsigned char*)insert_vectors[i], vector_len * sizeof(int));
		for (int j = 0; j < vector_len; j++) {
			insert_vectors[i][j] &= 0xFF;
		}

		if (vflag && i >= verbose_point) {
			gettimeofday(&end_interval_time, NULL);
			fprintf(stderr, "\rGenerating insert set... %.2f - %.1f ops/sec", (double)(i + 1) / num_inserts, 1000000. * num_inserts / NUM_VERBOSE_INTERVALS / (end_interval_time.tv_sec - start_interval_time.tv_sec + 1000000 * (end_interval_time.tv_usec - start_interval_time.tv_usec)));
			verbose_point = num_inserts * (++verbose_interval) / NUM_VERBOSE_INTERVALS - 1;
			gettimeofday(&start_interval_time, NULL);
		}
	}
	fprintf(stderr, "Done\n");

	fprintf(stderr, "Performing inserts... ");
	gettimeofday(&start_time, NULL);
	for (int i = 0; i < sizeof(insert_vectors)/sizeof(insert_vectors[0]); i++) {
		int ret = inverted_index_insert(&inv_index, insert_vectors[i], i);
	}
	gettimeofday(&end_time, NULL);
	fprintf(stderr, "Done\n");

	int *query_vectors = new int[num_queries * vector_len];
	RAND_bytes((unsigned char*)query_vectors[0], num_queries * vector_len * sizeof(int));
	std::vector<inverted_value> k_top;

	fprintf(stderr, "Performing queries... ");
	gettimeofday(&start_time, NULL);
	for (int i = 0; i < num_queries; i++) {
		k_top = inverted_index_query(&inv_index, &(query_vectors[i]), 3);
	}
	gettimeofday(&end_time, NULL);
	fprintf(stderr, "Done\n");
	printf("Query throughput: %.2f ops/sec\n", 1000000. * num_queries / (end_time.tv_sec - start_time.tv_sec + 1000000 * (end_time.tv_usec - start_time.tv_usec)));

	/*for (int i = 0; i < k_top.size(); i++) {
		printf("%lu\t%lu\n", k_top[i].id, k_top[i].value);
	}*/

	inverted_index_free(&inv_index);

	printf("Test completed\n");
	return 0;
}
