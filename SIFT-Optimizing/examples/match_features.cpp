#include <iostream>
#include <string>
#include <chrono>
#include <pthread.h>
#include "image.hpp"
#include "sift.hpp"

struct ThreadData {
    Image* image;
    std::vector<sift::Keypoint>* keypoints;
    int gpu_id;
};

void* process_image(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    // Set GPU for this thread
    // hipSetDevice(data->gpu_id);

    // Perform keypoint detection and descriptor computation
    *(data->keypoints) = sift::find_keypoints_and_descriptors(*(data->image), data->gpu_id);

    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 3) {
        std::cerr << "Usage: ./match_features a.jpg b.jpg (or .png)\n";
        return 0;
    }

    Image a(argv[1]), b(argv[2]);
    a = a.channels == 1 ? a : rgb_to_grayscale(a);
    b = b.channels == 1 ? b : rgb_to_grayscale(b);

    auto start = std::chrono::high_resolution_clock::now();

    // Keypoints for both images
    std::vector<sift::Keypoint> kps_a, kps_b;

    // Thread data for each image
    ThreadData data_a = { &a, &kps_a, 0 }; // GPU 0
    ThreadData data_b = { &b, &kps_b, 1 }; // GPU 1

    // Create threads
    pthread_t thread_a, thread_b;
    pthread_create(&thread_a, nullptr, process_image, &data_a);
    pthread_create(&thread_b, nullptr, process_image, &data_b);

    // Wait for threads to finish
    pthread_join(thread_a, nullptr);
    pthread_join(thread_b, nullptr);

    auto start_compute = std::chrono::high_resolution_clock::now();

    // Match features
    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps_a, kps_b);

    auto end_compute = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    auto compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute);
    std::cout << std::endl << "Matching time: " << compute_duration.count() << " ms" << std::endl;

    compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << std::endl << "Total executing time: " << compute_duration.count() << " ms" << std::endl;

    // Draw and save result
    Image result = sift::draw_matches(a, b, kps_a, kps_b, matches);
    result.save("result.jpg");

    std::cout << "Found " << matches.size() << " feature matches. Output image is saved as result.jpg\n";
    return 0;
}

