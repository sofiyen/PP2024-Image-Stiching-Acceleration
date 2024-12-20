#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"
#include <chrono>

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 2) {
        std::cerr << "Usage: ./find_keypoints input.jpg (or .png)\n";
        return 0;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    Image img(argv[1]);
    img =  img.channels == 1 ? img : rgb_to_grayscale(img);

    std::cout << "Image size: " << img.width << " x " << img.height << std::endl;

    auto start_compute = std::chrono::high_resolution_clock::now();
    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(img);
    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute);
    std::cout << "Finding keypoint time: " << compute_duration.count() << " ms" << std::endl;



    Image result = sift::draw_keypoints(img, kps);
    result.save("result.jpg");

    auto end_time = std::chrono::high_resolution_clock::now();

    // 計算執行時間
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;


    std::cout << "Found " << kps.size() << " keypoints. Output image is saved as result.jpg\n";
    return 0;
}
