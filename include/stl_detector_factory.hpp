#ifndef BGM_STLD_STL_DETECTOR_FACTORY_HPP_
#define BGM_STLD_STL_DETECTOR_FACTORY_HPP_

#include "detection_type.hpp"
#include "detector_for_roi.hpp"
#include "yolov2_caffe_detector.hpp"
#include "yolov2_caffe_decoder.hpp"
#include "nms_rect.hpp"
#include "nms_distance_rect.hpp"
#include "threshold_and_nms.hpp"

#include <memory>

namespace bgm
{

typedef DetectionRect<float, float> TL;
typedef DetectorForROI<TL> STLDetector;

template <typename AnchorInIterT,
          typename AnchorWInIterT,
          typename ROIInIterT>
std::shared_ptr<STLDetector> STLDetectorFactory(
    const std::string& caffe_net, const std::string& caffe_model,
    bool use_gpu,
    int num_class,
    const AnchorInIterT& anchor_beg, 
    const AnchorInIterT& anchor_end,
    const AnchorWInIterT& anchor_weight_beg,
    const AnchorWInIterT& anchor_weight_end,
    const cv::Size2f& grid_cell_size,
    const ROIInIterT& roi_beg, const ROIInIterT& roi_end,
    float conf_thresh, float nms_overlap, float nms_dist_mult);

std::shared_ptr<STLDetector> BoschDetectorFactory(
    const std::string& caffe_net, const std::string& caffe_model,
    bool use_gpu);





// definitions
template <typename AnchorInIterT, 
          typename AnchorWInIterT,
          typename ROIInIterT>
std::shared_ptr<STLDetector> STLDetectorFactory(
    const std::string& caffe_net, const std::string& caffe_model,
    bool use_gpu,
    int num_class,
    const AnchorInIterT& anchor_beg, AnchorInIterT& anchor_end,
    const AnchorWInIterT& anchor_weight_beg,
    const AnchorWInIterT& anchor_weight_end,
    const cv::Size2f& grid_cell_size,
    const ROIInIterT& roi_beg, const ROIInIterT& roi_end,
    float conf_thresh, float nms_overlap, float nms_dist_mult) {
  CaffeWrapper<float>* caffe = 
      new CaffeWrapper<float>(caffe_net, caffe_model, use_gpu);

  YOLOv2CaffeDecoder<float>* yolov2_decoder = 
      new YOLOv2CaffeDecoder<float>(num_class, anchor_beg, anchor_end,
                                    anchor_weight_beg, anchor_weight_end,
                                    grid_cell_size);

  YOLOV2CaffeDetector<float>* yolov2_detector = 
      new YOLOV2CaffeDetector<float>(caffe, yolov2_decoder);

  NMSRect<float, float>* nms = new NMSRect<float, float>(nms_overlap);
  NMSDistanceRect<float, float>* dist_nms =
      new NMSDistanceRect<float, float>(nms, nms_dist_mult);
  DetectionFilter<TL> filter = 
      new ThresholdAndNMS<TL>(conf_thresh, dist_nms);

  STLDetector* stl_detector = 
      new STLDetector(yolov2_detector, roi_beg, roi_end, filter, true);

  return std::shared_ptr<STLDetector>(stl_detector);
}

std::shared_ptr<STLDetector> BoschDetectorFactory(
    const std::string& caffe_net, const std::string& caffe_model,
    bool use_gpu) {
  cv::Rect2f anchor[4];
  cv::Rect2f* anchor_ptr = anchor;

  float anchor_weight[4];
  float* anchor_w_ptr = anchor_weight;

  cv::Size2f grid_cell_size(4, 4);

  cv::Rect roi[12];
  cv::Rect* roi_ptr;

  float conf_thresh;
  float nms_overlap = 0.1;
  float nms_range = 3;

  return STLDetectorFactory(caffe_net, caffe_model, use_gpu,
                            4, anchor_ptr, anchor_ptr + 4,
                            anchor_w_ptr, anchor_w_ptr + 4,
                            grid_cell_size, roi_ptr, roi_ptr + 12,
                            conf_thresh, nms_overlap, nms_range);
}

} // namespace bgm

#endif // !BGM_STLD_STL_DETECTOR_FACTORY_HPP_
