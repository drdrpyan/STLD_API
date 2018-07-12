#ifndef BGM_STLD_STL_DETECTOR_HPP_
#define BGM_STLD_STL_DETECTOR_HPP_

#include "detection_type.hpp"
#include "detector.hpp"
#include "nms.hpp"

#include <opencv2/core.hpp>

#include <glog/logging.h>

#include <memory>

namespace bgm
{

class STLDetector
{
  typedef DetectionRect<float, float> TL;

 public:
  void Detect(const cv::Mat& img, std::vector<TL>* result);
  template <typename InIterT>
  void set_offset(const InIterT& beg, const InIterT& end);
  void set_patch_size(const cv::Size& size);
  void set_patch_size(int w, int h);

 private:
  void ExtractPatches(const cv::Mat& in_mat,
                      std::vector<cv::Mat>* patches) const;
  void MovePatchResults(std::vector<std::vector<TL> >& patch_results) const;
  void PickStrongConfs(const std::vector<std::vector<TL> >& patch_results,
                       std::vector<TL>* strong_conf) const;

  std::shared_ptr<Detector<TL> > detector_;
  std::shared_ptr<NMS<TL> > nms_;

  std::vector<cv::Point> offset_;
  cv::Size patch_size_;
  float conf_threshold_;

}; // class STLDetector

// inline functions
inline void STLDetector::set_patch_size(const cv::Size& size) {
  CHECK_GT(size.width, 0);
  CHECK_GT(size.height, 0);
  patch_size_ = size;
}

inline void STLDetector::set_patch_size(int w, int h) {
  CHECK_GT(w, 0);
  CHECK_GT(h, 0);
  patch_size_ = cv::Size(w, h);
}

// template functions
template <typename InIterT>
void STLDetector::set_offset(const InIterT& beg, const InIterT& end) {
  offset_.assign(beg, end);
}

} // namespace bgm

#endif // !BGM_STLD_STLD_DETECTOR_HPP_
