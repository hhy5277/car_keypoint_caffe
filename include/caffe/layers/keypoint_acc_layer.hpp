#ifndef CAFFE_ACCURACY_LAYER_HPP_
#define CAFFE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class KeypointAccLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit KeypointAccLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Accuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_, channels_, height_, width_, hm_dims_;

  Dtype metric_;

  /// Keeps counts of the number of samples per class.
  Blob<Dtype> nums_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
