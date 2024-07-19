#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#include <unordered_set>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

int main(int argc, char **argv) {
    int hole = atoi(argv[1]);

    uint32_t n, d;
    std::string path = "data/graph/base_1M_64_128";
    std::ifstream reader(path);

    reader.read((char*)&n, sizeof(uint32_t));
    reader.read((char*)&d, sizeof(uint32_t));

    auto sizes = parlay::sequence<uint32_t>::uninitialized(n);
    reader.read((char*)&sizes[0], n * sizeof(uint32_t));

    parlay::sequence<parlay::sequence<uint32_t>> edges;
    for (int i = 0; i < n; i++) {
        auto edge_list = parlay::sequence<uint32_t>::uninitialized(sizes[i]);
        reader.read((char*)&edge_list[0], sizes[i] * sizeof(uint32_t));
        edges.push_back(edge_list);
    }

    std::unordered_set<uint32_t> visited;
    std::unordered_set<uint32_t> fringe;
    visited.insert(0);
    fringe.insert(0);

    std::cout << "Creating donut from level " << hole << "..." << std::endl;
    for (int i = 0; i < hole; i++) {
        std::unordered_set<uint32_t> new_fringe;
        for (uint32_t node : fringe) {
            for (uint32_t neighbor : edges[node]) {
                if (visited.count(neighbor) == 0) {
                    visited.insert(neighbor);
                    new_fringe.insert(neighbor);
                }
            }
        }
        fringe = new_fringe;
    }

    fringe.clear();
    while (true) {
        uint32_t start = rand() % n;
        if (visited.count(start) == 0) {
            fringe.insert(start);
            break;
        }
    }

    std::cout << "Searching BFS tree..." << std::endl << "Level 0 has 1 node" << std::endl;
    int level = 0;
    while (fringe.size() > 0) {
        std::unordered_set<uint32_t> new_fringe;
        for (uint32_t node : fringe) {
            for (uint32_t neighbor : edges[node]) {
                if (visited.count(neighbor) == 0) {
                    visited.insert(neighbor);
                    new_fringe.insert(neighbor);
                }
            }
        }
        std::cout << "Level " << (++level) << " has " << new_fringe.size() << " nodes" << std::endl;
        fringe = new_fringe;
    }
    std::cout << "There are " << n - visited.size() << " unreachable nodes" << std::endl;

    return 0;
}
