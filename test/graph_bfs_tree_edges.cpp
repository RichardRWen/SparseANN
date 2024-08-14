#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#include <unordered_set>
#include <unordered_map>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

int main(int argc, char **argv) {
    std::string path = "data/graph/base_1M_64_128";
    if (argc > 1) {
        path = argv[1];
    }
    std::ifstream reader(path);

    uint32_t n, d;
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

    std::unordered_map<uint32_t, uint32_t> level_map;
    std::unordered_set<uint32_t> visited;
    std::vector<std::unordered_set<uint32_t>> levels;
    visited.insert(0);
    level_map[0] = 0;
    levels.push_back(std::unordered_set<uint32_t>());
    levels[0].insert(0);

    std::cout << "Searching BFS tree..." << std::endl << "Level 0 has 1 node" << std::endl;
    int level = 0;
    while (levels[levels.size() - 1].size() > 0) {
        levels.push_back(std::unordered_set<uint32_t>());
        for (uint32_t node : levels[levels.size() - 2]) {
            for (uint32_t neighbor : edges[node]) {
                if (visited.count(neighbor) == 0) {
                    visited.insert(neighbor);
                    levels[levels.size() - 1].insert(neighbor);
                    level_map[neighbor] = levels.size() - 1;
                }
            }
        }
        std::cout << "Level " << (levels.size() - 1) << " has " << levels[levels.size() - 1].size() << " nodes" << std::endl;
    }
    std::cout << "There are " << n - visited.size() << " unreachable nodes" << std::endl;

    return 0;
}
