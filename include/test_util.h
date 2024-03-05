#include <iostream>
#include <functional>
#include <chrono>

void time_function(std::string caption, std::function<void()> func) {
	std::cout << caption << "...  " << std::flush;
	auto start = std::chrono::high_resolution_clock::now();
	
	func();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "done in " << duration.count() / 1000. << " seconds." << std::endl;
}
