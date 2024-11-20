#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <utility>
#include <cstdint>

// Custom hash function for std::pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ (std::hash<T2>()(pair.second) << 1);
    }
};

using Point = std::vector<float>;
using CellKey = std::pair<int, int>;
// ClusterMap now maps from cell key to a vector of cluster indices
using ClusterMap = std::unordered_map<CellKey, std::vector<size_t>, pair_hash>;

// Function to compute the cell key for a point
CellKey getCellKey(const Point& point, float cell_size = 0.02) {
    return {static_cast<int>(point[0] / cell_size), static_cast<int>(point[1] / cell_size)};
}

// Check if two points are close enough in each dimension
bool isClose(const Point& p1, const Point& p2, float threshold = 0.02) {
    for (size_t i = 0; i < p1.size(); ++i) {
        if (std::abs(p1[i] - p2[i]) > threshold) {
            return false;
        }
    }
    return true;
}

// Function to identify clusters using point IDs instead of points directly
void identifyClusters(const std::vector<Point>& points, std::vector<std::vector<size_t>>& clusters, ClusterMap& cluster_map) {
    float cell_size = 0.02;
    
    for (size_t point_id = 0; point_id < points.size(); ++point_id) {
        const Point& point = points[point_id];
        CellKey key = getCellKey(point, cell_size);
        
        bool found_cluster = false;
        // Check the 3x3 neighborhood around the cell
        for (int dx = -1; dx <= 1 && !found_cluster; ++dx) {
            for (int dy = -1; dy <= 1 && !found_cluster; ++dy) {
                CellKey neighbor_key = {key.first + dx, key.second + dy};
                
                if (cluster_map.find(neighbor_key) != cluster_map.end()) {
                    for (size_t cluster_id : cluster_map[neighbor_key]) {
                        const Point& center = points[clusters[cluster_id][0]];  // Access the center using point ID
                        if (isClose(point, center)) {
                            clusters[cluster_id].push_back(point_id);
                            found_cluster = true;
                            break;
                        }
                    }
                }
            }
        }
        
        // If no nearby cluster is found, create a new cluster with this point as the center
        if (!found_cluster) {
            clusters.push_back({point_id});  // Start new cluster with this point ID as the center
            size_t new_cluster_id = clusters.size() - 1;
            cluster_map[key].push_back(new_cluster_id);  // Map cell to this new cluster ID
        }
    }
}

// Load points from a binary file
std::vector<Point> load_points_from_file(const std::string& filename) {
    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        return {};
    }

    uint32_t num_vecs, num_dims;
    reader.read((char*)(&num_vecs), sizeof(uint32_t));
    reader.read((char*)(&num_dims), sizeof(uint32_t));

    std::vector<Point> point_set(num_vecs, std::vector<float>(num_dims));
    for (size_t i = 0; i < num_vecs; ++i) {
        reader.read((char*)point_set[i].data(), num_dims * sizeof(float));
    }

    reader.close();
    return point_set;
}

// Save clusters to a binary file
void save_clusters_to_file(const std::string& filename, const std::vector<Point>& points,
                           const std::vector<std::vector<size_t>>& clusters, uint32_t dimensions) {
    std::ofstream writer(filename, std::ios::binary);
    if (!writer.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        return;
    }

    uint32_t num_clusters = clusters.size();
    writer.write((char*)(&num_clusters), sizeof(uint32_t));
    writer.write((char*)(&dimensions), sizeof(uint32_t));

    for (const auto& cluster : clusters) {
        uint32_t cluster_size = cluster.size();
        const Point& center = points[cluster[0]];  // First point in the cluster as the center
        writer.write((char*)(&cluster_size), sizeof(uint32_t));
        writer.write((char*)center.data(), dimensions * sizeof(float));
    }

    writer.close();
}

int main() {
    // Load points from file
    //std::string filename = "/ssd1/trubel/ANNbench_/data/ba_1M_100/ba_1M_100.fbin";
    //std::string filename = "/ssd1/trubel/ANNbench_/data/ba_1M_100/base.fbin";
    std::string filename = "ba_1M_100.fbin";
    std::vector<Point> points = load_points_from_file(filename);
    std::cout << "Total points loaded: " << points.size() << std::endl;

    if (points.empty()) {
        std::cerr << "Error loading points from file.\n";
        return 1;
    }

    // Initialize clusters and cluster map
    std::vector<std::vector<size_t>> clusters;  // Clusters store point IDs
    ClusterMap cluster_map;
    identifyClusters(points, clusters, cluster_map);

    std::cout << "Total clusters found: " << clusters.size() << std::endl;

    // Output each cluster's points
    for (size_t i = 0; i < clusters.size() && i < 5; ++i) {
        std::cout << "Cluster " << i << " (Center ID: " << clusters[i][0] << ", Num points: " << clusters[i].size() << "):\n";
        std::cout << "  Points:\n";
        size_t j = 0;
        for (size_t point_id : clusters[i]) {
            if (j++ > 10) {
                std::cout << "..." << std::endl;
                break;
            }
            const Point& point = points[point_id];
            std::cout << "    ";
            size_t k = 0;
            for (float coord : point) {
                if (k++ > 10) {
                    std::cout << "...";
                    break;
                }
                std::cout << coord << " ";
            }
            std::cout << "\n";
        }
    }

    // Save clusters to an output file
    std::string output_filename = "cluster_data.bin";  // Adjust output filename as needed
    uint32_t dimensions = points[0].size();  // Dimensionality of points
    save_clusters_to_file(output_filename, points, clusters, dimensions);

    return 0;
}

