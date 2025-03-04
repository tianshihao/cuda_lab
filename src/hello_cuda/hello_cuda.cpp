#include "hello_cuda.h"

#include <opencv2/opencv.hpp>

namespace cuda_image_lab::hello_cuda {
void ShowLena() {
  cv::Mat img{cv::imread(
      "/home/tianshihao/source/repos/cuda_lab/images/Lenna_(test_image).png")};
  cv::imshow("lena", img);
  cv::waitKey(0);
}
void ShowImage() {
  cv::Mat img{cv::Mat::zeros(512, 1280, CV_8UC3)};
  float rescale_factor{255.0f};
  cv::Point3f color_mean_for_padding{0.48145466, 0.4578275, 0.40821073};
  cv::Point3f color_mean{0.5f, 0.5f, 0.5f};
  cv::Point3f color_scale{0.5f, 0.5f, 0.5f};

  std::array<float, 3> padding_value;
  padding_value[0] =
      (static_cast<int>(color_mean_for_padding.x * rescale_factor) /
           rescale_factor -
       color_mean.x) /
      color_scale.x;
  padding_value[1] =
      (static_cast<int>(color_mean_for_padding.y * rescale_factor) /
           rescale_factor -
       color_mean.y) /
      color_scale.y;
  padding_value[2] =
      (static_cast<int>(color_mean_for_padding.z * rescale_factor) /
           rescale_factor -
       color_mean.z) /
      color_scale.z;

  for (int row{0}; row < img.rows; ++row) {
    for (int col{0}; col < img.cols; ++col) {
      // Set RGB
      img.at<cv::Vec3b>(row, col)[0] = padding_value[0] * rescale_factor;
      img.at<cv::Vec3b>(row, col)[1] = padding_value[1] * rescale_factor;
      img.at<cv::Vec3b>(row, col)[2] = padding_value[2] * rescale_factor;
    }
  }

  cv::imshow("image", img);
  cv::waitKey(0);
}
}  // namespace cuda_image_lab::hello_cuda