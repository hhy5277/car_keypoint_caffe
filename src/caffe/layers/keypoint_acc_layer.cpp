#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/keypoint_acc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void KeypointAccLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        metric_ = this->layer_param_.keypoint_acc_param().metric();
    }

    template<typename Dtype>
    void KeypointAccLayer<Dtype>::Reshape(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        CHECK_EQ(bottom[0]->num_axes(), 4) << "bottom[0]->num_axes() must be 4.";
        CHECK_EQ(bottom[1]->num_axes(), 4) << "bottom[1]->num_axes() must be 4.";
        CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "Inputs must have the same num.";
        CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "Inputs must have the same channels.";
        CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Inputs must have the same height.";
        CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Inputs must have the same width.";
        num_ = bottom[0]->shape(0);
        channels_ = bottom[0]->shape(1);
        height_ = bottom[0]->shape(2);
        width_ = bottom[0]->shape(3);
        hm_dims_ = height_ * width_;
        vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
        top[0]->Reshape(top_shape);
        if (top.size() > 1)
        {
            // Per-class accuracy is a vector; 1 axes.
            vector<int> points_num(1);
            points_num[0] = channels_;
            top[1]->Reshape(points_num);
            nums_buffer_.Reshape(points_num);
        }
    }

    template<typename Dtype>
    void KeypointAccLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top) {
        Dtype accuracy = 0;
        const Dtype *bottom_data = bottom[0]->cpu_data();
        const Dtype *bottom_label = bottom[1]->cpu_data();
//  const int dim = bottom[0]->height()*bottom[0]->width();
//  const int num_labels = bottom[0]->shape(label_axis_);
        if (top.size() > 1)
        {
            caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
            caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
        }
        int count = 0;
        for (int i = 0; i < num_; ++i)
        {
            for (int j = 0; j < channels_; ++j)
            {
                if (std::abs(bottom_label[(channels_ * i + j) * hm_dims_] + 1) > 1e-6) //非隐藏点
                {
                    int offset = (channels_ * i + j) * hm_dims_;
                    Dtype max_prob = bottom_data[offset];
                    int max_prob_idx = 0;
                    Dtype max_gt = bottom_label[offset];
                    int max_gt_idx = 0;
                    for (int k = 0; k < hm_dims_; ++k)
                    {
                        int idx = offset + k;
                        if (bottom_data[idx] > max_prob)
                        {
                            max_prob = bottom_data[idx];
                            max_prob_idx = k;
                        }
                        if (bottom_label[idx] > max_gt)
                        {
                            max_gt = bottom_label[idx];
                            max_gt_idx = k;
                        }
                    }
                    //标准化
                    Dtype pred_x = static_cast<Dtype>(max_prob_idx % width_) / width_;
                    Dtype pred_y = static_cast<Dtype>(max_prob_idx / width_) / height_;
                    Dtype gt_x = static_cast<Dtype>(max_gt_idx % width_) / width_;
                    Dtype gt_y = static_cast<Dtype>(max_gt_idx / width_) / height_;
                    if (std::pow(pred_x - gt_x, 2) + std::pow(pred_y - gt_y, 2) < metric_ * metric_)
                    {
                        if (top.size() > 1)
                        {
                            ++top[1]->mutable_cpu_data()[j]; //预测准确
                        }
                        ++accuracy;
                    }
                    if (top.size() > 1)
                    {
                        ++nums_buffer_.mutable_cpu_data()[j]; //关键点gt可见点
                    }
                    ++count; //所有点可见
                }
            }
        }
        LOG(INFO) << "Accuracy: " << accuracy<<"  (accuracy / count):"<<accuracy<<"/"<<count<<std::endl;
        top[0]->mutable_cpu_data()[0] = (count == 0) ? 0 : (accuracy / count);
        if (top.size() > 1)
        {
            for (int i = 0; i < top[1]->count(); ++i)
            {
                top[1]->mutable_cpu_data()[i] =
                        nums_buffer_.cpu_data()[i] == 0 ? 0
                                                        : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
            }
        }
        // Accuracy layer should not be used as a loss function.
    }

#ifdef CPU_ONLY
    STUB_GPU(AccuracyLayer);
#endif

    INSTANTIATE_CLASS(KeypointAccLayer);

    REGISTER_LAYER_CLASS(KeypointAcc);

}  // namespace caffe
