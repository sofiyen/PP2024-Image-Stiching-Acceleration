#include <iostream>
#include <string>
#include <chrono> // Include chrono for timing
#include <opencv2/opencv.hpp>
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

class ImageStitcher
{
private:
    cv::Mat H_mean;
    int H_count = 0;
    const int BLEND_WIDTH = 150;  // 增加混合寬度
    struct MaxRect {
        cv::Point top_left;
        int width;
        int height;
    };

    MaxRect find_largest_rectangle(const cv::Mat& mask) {
        cv::Mat integral;
        cv::integral(mask, integral, CV_32S);
        
        MaxRect max_rect{cv::Point(0,0), 0, 0};
        int max_area = 0;
        
        // For each possible height
        for(int h1 = 0; h1 < mask.rows; h1++) {
            for(int h2 = h1; h2 < mask.rows; h2++) {
                std::vector<int> row_sum(mask.cols, 0);
                
                // Calculate row sums for the current height range
                for(int w = 0; w < mask.cols; w++) {
                    int sum = integral.at<int>(h2+1, w+1) - 
                             integral.at<int>(h1, w+1) - 
                             integral.at<int>(h2+1, w) + 
                             integral.at<int>(h1, w);
                    
                    // If the entire column in this height range is valid (non-zero)
                    row_sum[w] = (sum == (h2-h1+1) * 255) ? 1 : 0;
                }
                
                // Find longest sequence of 1s
                int curr_start = 0;
                int curr_length = 0;
                
                for(int w = 0; w < mask.cols; w++) {
                    if(row_sum[w] == 1) {
                        curr_length++;
                        int area = curr_length * (h2-h1+1);
                        if(area > max_area) {
                            max_area = area;
                            max_rect.top_left = cv::Point(w-curr_length+1, h1);
                            max_rect.width = curr_length;
                            max_rect.height = h2-h1+1;
                        }
                    } else {
                        curr_length = 0;
                    }
                }
            }
        }
        
        return max_rect;
    }
    cv::Rect find_content_bounds(const cv::Mat& img) {
        cv::Mat gray;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = img.clone();
        }
        
        // Find non-zero pixels
        cv::Mat non_zero;
        cv::findNonZero(gray, non_zero);
        
        // Get bounding rectangle
        return cv::boundingRect(non_zero);
    }
    std::vector<cv::Point2f> convert_keypoints_to_points(
        const std::vector<sift::Keypoint> &keypoints,
        const std::vector<std::pair<int, int>> &matches,
        bool is_image_a)
    {
        std::vector<cv::Point2f> points;
        for (const auto &match : matches)
        {
            int idx = is_image_a ? match.first : match.second;
            points.emplace_back(keypoints[idx].x, keypoints[idx].y);
        }
        return points;
    }

    struct PanoramaSize
    {
        cv::Size size;
        cv::Point offset;
    };

    struct ColorStats {
        cv::Vec3f mean;
        cv::Vec3f stddev;
    };

    ColorStats calculate_color_stats(const cv::Mat& img, const cv::Mat& mask) {
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F);
        cv::Scalar mean, stddev;
        cv::meanStdDev(img_float, mean, stddev, mask);
        return {cv::Vec3f(mean[0], mean[1], mean[2]), 
                cv::Vec3f(stddev[0], stddev[1], stddev[2])};
    }

    cv::Mat create_distance_weight_mask(const cv::Mat& mask, int blend_width) {
        cv::Mat dist;
        cv::distanceTransform(mask, dist, cv::DIST_L2, 3);

        cv::Mat weight_mask(dist.size(), CV_32F);
        for(int y = 0; y < dist.rows; y++) {
            for(int x = 0; x < dist.cols; x++) {
                float d = dist.at<float>(y, x);
                if(d < blend_width) {
                    weight_mask.at<float>(y, x) = d / blend_width;
                } else {
                    weight_mask.at<float>(y, x) = 1.0f;
                }
            }
        }
        return weight_mask;
    }

    void adjust_color_balance(cv::Mat& img_to_adjust, const cv::Mat& mask_to_adjust,
                        const cv::Mat& target_img, const cv::Mat& target_mask) {
        // 計算重疊區域
        cv::Mat overlap;
        cv::bitwise_and(mask_to_adjust, target_mask, overlap);

        // 擴展重疊區域以獲得更平滑的過渡
        cv::Mat expanded_overlap;
        cv::dilate(overlap, expanded_overlap, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(51, 51)));

        // 計算顏色統計（使用擴展後的區域）
        ColorStats stats1 = calculate_color_stats(img_to_adjust, expanded_overlap);
        ColorStats stats2 = calculate_color_stats(target_img, expanded_overlap);

        // 創建平滑的權重遮罩
        cv::Mat weight_mask;
        cv::distanceTransform(expanded_overlap, weight_mask, cv::DIST_L2, 5);
        cv::normalize(weight_mask, weight_mask, 0, 1, cv::NORM_MINMAX);
        // cv::GaussianBlur(weight_mask, weight_mask, cv::Size(51, 51), 25);
        // cv::GaussianBlur(weight_mask, weight_mask, cv::Size(81, 81), 35);

        // 計算兩張圖片的整體亮度差異
        cv::Scalar mean1 = cv::mean(img_to_adjust, expanded_overlap);
        cv::Scalar mean2 = cv::mean(target_img, expanded_overlap);
        float brightness_ratio = (mean2[0] + mean2[1] + mean2[2]) / 
                            (mean1[0] + mean1[1] + mean1[2] + 1e-5);

        // 應用顏色調整
        for(int y = 0; y < img_to_adjust.rows; y++) {
            for(int x = 0; x < img_to_adjust.cols; x++) {
                if(mask_to_adjust.at<uchar>(y, x) > 0) {
                    float w = weight_mask.at<float>(y, x);
                    cv::Vec3b& pixel = img_to_adjust.at<cv::Vec3b>(y, x);

                    // 使用平滑的 sigmoid 函數進行權重調整
                    // float sigmoid_w = 1.0f / (1.0f + std::exp(-(w - 0.5f) * 8.0f));
                    // 將8.0f改小，例如改為4.0f
                    float sigmoid_w = 1.0f / (1.0f + std::exp(-(w - 0.5f) * 4.0f));
                    
                    for(int c = 0; c < 3; c++) {
                        // 標準化顏色調整
                        float normalized = (pixel[c] - stats1.mean[c]) / (stats1.stddev[c] + 1e-5);
                        float adjusted = normalized * stats2.stddev[c] + stats2.mean[c];
                        
                        // 亮度調整
                        float brightness_adjusted = pixel[c] * brightness_ratio;
                        
                        // 結合兩種調整並應用平滑過渡
                        float t = sigmoid_w;
                        float smooth_w = t * t * (3 - 2 * t);  // smoothstep 函數
                        
                        float final_value = brightness_adjusted * (1.0f - smooth_w) + 
                                        adjusted * smooth_w;
                                        
                        pixel[c] = cv::saturate_cast<uchar>(final_value);
                    }
                }
            }
        }

        // 最後對調整區域進行平滑處理
        cv::Mat adjusted_region = img_to_adjust.clone();
        // cv::GaussianBlur(adjusted_region, adjusted_region, cv::Size(5, 5), 0);
        
        for(int y = 0; y < img_to_adjust.rows; y++) {
            for(int x = 0; x < img_to_adjust.cols; x++) {
                if(expanded_overlap.at<uchar>(y, x) > 0) {
                    float w = weight_mask.at<float>(y, x);
                    // 在過渡區域使用更強的模糊效果
                    float blend_factor = 0.5f - std::abs(w - 0.5f);
                    blend_factor = std::pow(blend_factor * 2.0f, 2.0f);
                    
                    if(blend_factor > 0) {
                        cv::Vec3b& pixel = img_to_adjust.at<cv::Vec3b>(y, x);
                        cv::Vec3b blurred = adjusted_region.at<cv::Vec3b>(y, x);
                        
                        for(int c = 0; c < 3; c++) {
                            pixel[c] = cv::saturate_cast<uchar>(
                                pixel[c] * (1.0f - blend_factor) + 
                                blurred[c] * blend_factor
                            );
                        }
                    }
                }
            }
        }
    }

    cv::Mat blend_images(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask1, const cv::Mat& mask2) {
        cv::Mat result = img2.clone();
        cv::Mat overlap;
        cv::bitwise_and(mask1, mask2, overlap);

        cv::Mat dist_weight = create_distance_weight_mask(overlap, BLEND_WIDTH);

        for(int y = 0; y < result.rows; y++) {
            for(int x = 0; x < result.cols; x++) {
                if(overlap.at<uchar>(y, x) > 0) {
                    float w = dist_weight.at<float>(y, x);
                    cv::Vec3b color1 = img1.at<cv::Vec3b>(y, x);
                    cv::Vec3b color2 = img2.at<cv::Vec3b>(y, x);
                    
                    // 使用平方根權重以獲得更平滑的過渡
                    float blend_weight = std::sqrt(w);
                    result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        cv::saturate_cast<uchar>(color1[0] * blend_weight + color2[0] * (1-blend_weight)),
                        cv::saturate_cast<uchar>(color1[1] * blend_weight + color2[1] * (1-blend_weight)),
                        cv::saturate_cast<uchar>(color1[2] * blend_weight + color2[2] * (1-blend_weight))
                    );
                }
                else if(mask1.at<uchar>(y, x) > 0) {
                    result.at<cv::Vec3b>(y, x) = img1.at<cv::Vec3b>(y, x);
                }
            }
        }
        return result;
    }

    PanoramaSize calculate_panorama_size(const cv::Mat &img_a, const cv::Mat &img_b, const cv::Mat &H)
    {
        std::vector<cv::Point2f> corners_a(4);
        corners_a[0] = cv::Point2f(0, 0);
        corners_a[1] = cv::Point2f(img_a.cols, 0);
        corners_a[2] = cv::Point2f(img_a.cols, img_a.rows);
        corners_a[3] = cv::Point2f(0, img_a.rows);

        std::vector<cv::Point2f> warped_corners_a;
        cv::perspectiveTransform(corners_a, warped_corners_a, H);

        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();

        for (const auto &corner : warped_corners_a)
        {
            min_x = std::min(min_x, corner.x);
            min_y = std::min(min_y, corner.y);
            max_x = std::max(max_x, corner.x);
            max_y = std::max(max_y, corner.y);
        }

        min_x = std::min(min_x, 0.0f);
        min_y = std::min(min_y, 0.0f);
        max_x = std::max(max_x, static_cast<float>(img_b.cols));
        max_y = std::max(max_y, static_cast<float>(img_b.rows));

        cv::Point offset(-std::floor(min_x), -std::floor(min_y));
        int width = std::ceil(max_x - min_x);
        int height = std::ceil(max_y - min_y);

        return {cv::Size(width, height), offset};
    }

public:
    ImageStitcher() : H_count(0) {}

    cv::Mat stitch_images(const std::string &image_a_path, const std::string &image_b_path, const cv::Mat &H) {
        cv::Mat img_a = cv::imread(image_a_path);
        cv::Mat img_b = cv::imread(image_b_path);

        if (img_a.empty() || img_b.empty()) {
            throw std::runtime_error("Failed to load images");
        }

        // Calculate panorama size and offset
        PanoramaSize pano = calculate_panorama_size(img_a, img_b, H);

        // Create transformation matrix
        cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, pano.offset.x,
                     0, 1, pano.offset.y,
                     0, 0, 1);
        cv::Mat H_final = T * H;

        // Warp image A and its mask
        cv::Mat warped_a;
        cv::warpPerspective(img_a, warped_a, H_final, pano.size);
        cv::Mat mask_a;
        cv::Mat temp = cv::Mat::ones(img_a.size(), CV_8UC1) * 255;
        cv::warpPerspective(temp, mask_a, H_final, pano.size);

        // Create result image and mask B
        cv::Mat result = cv::Mat(pano.size.height, pano.size.width, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat mask_b = cv::Mat::zeros(pano.size.height, pano.size.width, CV_8UC1);
        
        // Copy image B to result
        cv::Rect roi_b(pano.offset.x, pano.offset.y, img_b.cols, img_b.rows);
        img_b.copyTo(result(roi_b));
        mask_b(roi_b).setTo(255);

        // Adjust color balance
        adjust_color_balance(warped_a, mask_a, result, mask_b);

        // Blend images
        result = blend_images(warped_a, result, mask_a, mask_b);

        // Create a combined mask of both images
        cv::Mat combined_mask;
        cv::bitwise_or(mask_a, mask_b, combined_mask);

        // Find the largest valid rectangle
        MaxRect max_rect = find_largest_rectangle(combined_mask);
        
        // Crop the result to the largest valid rectangle
        cv::Rect crop_rect(max_rect.top_left.x, max_rect.top_left.y, 
                          max_rect.width, max_rect.height);
        result = result(crop_rect);

        return result;
    }



    cv::Mat calculate_homography(const std::vector<sift::Keypoint> &kps_a,
                               const std::vector<sift::Keypoint> &kps_b,
                               const std::vector<std::pair<int, int>> &matches)
    {
        if (matches.size() < 4)
        {
            throw std::runtime_error("Not enough matches to calculate homography");
        }

        std::vector<cv::Point2f> points_a = convert_keypoints_to_points(kps_a, matches, true);
        std::vector<cv::Point2f> points_b = convert_keypoints_to_points(kps_b, matches, false);

        std::vector<uchar> inliers;
        cv::Mat H = cv::findHomography(points_a, points_b, cv::RANSAC, 3.0, inliers, 2000, 0.995);

        if (H.empty())
        {
            throw std::runtime_error("Failed to calculate homography");
        }

        int inlier_count = std::count(inliers.begin(), inliers.end(), 1);
        double inlier_ratio = static_cast<double>(inlier_count) / matches.size();
        if (inlier_ratio < 0.3)
        {
            throw std::runtime_error("Poor homography: too few inliers");
        }

        H_count++;
        if (H_count == 1)
            H_mean = H.clone();
        else
            H_mean = 0.9 * H_mean + 0.1 * H;

        return H_mean;
    }
};


int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 3) {
        std::cerr << "Usage: ./match_features a.jpg b.jpg (or .png)\n";
        return 0;
    }

    try {
        // Load images and convert to grayscale
        auto start = std::chrono::high_resolution_clock::now();
        Image a(argv[1]), b(argv[2]);
        a = a.channels == 1 ? a : rgb_to_grayscale(a);
        b = b.channels == 1 ? b : rgb_to_grayscale(b);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Image loading and preprocessing took: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

        // Extract keypoints and descriptors
        start = std::chrono::high_resolution_clock::now();
        std::vector<sift::Keypoint> kps_a = sift::find_keypoints_and_descriptors(a);
        std::vector<sift::Keypoint> kps_b = sift::find_keypoints_and_descriptors(b);
        // Keypoints for both images
        // std::vector<sift::Keypoint> kps_a, kps_b;

        // // Thread data for each image
        // ThreadData data_a = { &a, &kps_a, 0 }; // GPU 0
        // ThreadData data_b = { &b, &kps_b, 1 }; // GPU 1

        // // Create threads
        // pthread_t thread_a, thread_b;
        // pthread_create(&thread_a, nullptr, process_image, &data_a);
        // pthread_create(&thread_b, nullptr, process_image, &data_b);

        // // Wait for threads to finish
        // pthread_join(thread_a, nullptr);
        // pthread_join(thread_b, nullptr);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Keypoints extraction took: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

        // Match features
        start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps_a, kps_b);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Feature matching took: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

        // Create stitcher and calculate homography
        ImageStitcher stitcher;
        std::cout << "Calculating homography...\n";
        cv::Mat H = stitcher.calculate_homography(kps_a, kps_b, matches);

        // Stitch images with direct overlay
        std::cout << "Overlaying images...\n";
        cv::Mat result = stitcher.stitch_images(argv[1], argv[2], H);

        // Save result
        std::cout << "Saving result...\n";
        cv::imwrite("overlaid_result.jpg", result);

        std::cout << "Image overlay completed successfully!\n";
        std::cout << "Found " << matches.size() << " feature matches.\n";
        std::cout << "Output image saved as overlaid_result.jpg\n";
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Total executing time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
