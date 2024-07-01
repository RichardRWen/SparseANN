#include <iostream>
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <parlay/internal/get_time.h>

#include "forward_index.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " [path to .csr file] [path to outfile] [desired number of vecs]" << std::endl;
        exit(0);
    }
    size_t desired_size = strtoull(argv[3], NULL, 10);

    parlay::internal::timer timer;
    timer.start();

    std::cout << "Reading from " << argv[1] << "... " << std::flush;
	forward_index<float> inserts = forward_index<float>(argv[1], "csr", desired_size);
    std::cout << "Done" << std::endl;
    std::cout << "Truncated to " << inserts.size() << " vectors of dimension " << inserts.dims << std::endl;

    std::cout << "Writing to " << argv[2] << "... " << std::flush;
    inserts.write_to_file(argv[2], "csr");
    std::cout << "Done" << std::endl;

    std::cout << "Took " << timer.next_time() << " seconds" << std::endl;
}
