import struct
import matplotlib.pyplot as plt

# Load clusters from the binary file
def load_clusters_from_file(filename):
    clusters = []
    with open(filename, "rb") as file:
        # Read the number of clusters and dimensions
        num_clusters, dimensions = struct.unpack("II", file.read(8))
        
        # Read each cluster's size and center
        for _ in range(num_clusters):
            cluster_size = struct.unpack("I", file.read(4))[0]
            center = struct.unpack(f"{dimensions}f", file.read(4 * dimensions))
            clusters.append((cluster_size, center))
    
    return clusters

# Plot the histogram of cluster sizes and save it as an image
def save_cluster_size_histogram(clusters, output_filename="cluster_size_histogram.png"):
    cluster_sizes = [size for size, _ in clusters]
    plt.figure(figsize=(10, 5))
    plt.hist(cluster_sizes, bins=20, edgecolor='black')
    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    plt.title("Histogram of Cluster Sizes")
    plt.savefig(output_filename)
    plt.close()  # Close the figure after saving to free up memory

# Plot cluster centers on the [0, 1] x [0, 1] box and save it as an image
def save_cluster_centers(clusters, output_filename="cluster_centers.png"):
    x_coords = [center[0] for _, center in clusters]
    y_coords = [center[1] for _, center in clusters]
    sizes = [size * 0.01 for size, _ in clusters]  # Scale size for visibility
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, s=sizes, alpha=0.5, edgecolor="black")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Cluster Centers")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(output_filename)
    plt.close()  # Close the figure after saving to free up memory

# Load the data and save the plots as images
filename = "../build/cluster_data.bin"  # Adjust filename as needed
clusters = load_clusters_from_file(filename)
save_cluster_size_histogram(clusters, "cluster_size_histogram.png")
save_cluster_centers(clusters, "cluster_centers.png")

