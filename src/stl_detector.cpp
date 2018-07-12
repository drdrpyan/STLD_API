#include "stl_detector.hpp"

#include <glog/logging.h>

namespace bgm
{

void STLDetector::Detect(const cv::Mat& img, 
                         std::vector<TL>* result) {
  CHECK(result);

  std::vector<cv::Mat> patches;
  ExtractPatches(img, &patches);

  std::vector<std::vector<TL> > patch_results;
  detector_->Detect(patches, false, &patch_results);

  MovePatchResults(patch_results);

  std::vector<TL> strong_confs;
  PickStrongConfs(patch_results, &strong_confs);

  nms_->nms(strong_confs, result);
}

void STLDetector::ExtractPatches(const cv::Mat& in_mat,
                                 std::vector<cv::Mat>* patches) const {
  CHECK(patches);
  patches->resize(offset_.size());
  for (int i = 0; i < offset_.size(); ++i) {
    (*patches)[i] = in_mat(cv::Rect(offset_[i], patch_size_));
  }
}

void STLDetector::MovePatchResults(
    std::vector<std::vector<TL> >& patch_results) const {
  for (int i = 0; i < patch_results.size(); ++i) {
    for (int j = 0; j < patch_results[i].size(); ++j) {
      cv::Rect_<float> rect = patch_results[i][j].rect();
      rect.x += offset_[i].x;
      rect.y += offset_[i].y;
      patch_results[i][j].set_rect(rect);
    }
  }
}

void STLDetector::PickStrongConfs(
    const std::vector<std::vector<TL> >& patch_results,
    std::vector<TL>* strong_conf) const {
  CHECK(strong_conf);

  strong_conf->clear();

  for (int i = 0; i < patch_results.size(); ++i) {
    for (int j = 0; j < patch_results[i].size(); ++j) {
      if (patch_results[i][j].conf() > conf_threshold_)
        strong_conf->push_back(patch_results[i][j]);
    }
  }
}

} // namespace bgm