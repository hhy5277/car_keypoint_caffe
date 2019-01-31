//
// Created by jiangshihong on 18-12-4.
//

#ifndef CAFFE_HEATMAP_DATA_LAYER_HPP
#define CAFFE_HEATMAP_DATA_LAYER_HPP


#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
    template <typename Dtype>
    class HeatmapDataLayer : public BasePrefetchingDataLayer<Dtype> {
    public:
        explicit HeatmapDataLayer(const LayerParameter& param)
                : BasePrefetchingDataLayer<Dtype>(param) {}
        virtual ~HeatmapDataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "HeatmapData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 2; }

    protected:
        shared_ptr<Caffe::RNG> prefetch_rng_;
        virtual void ShuffleImages();
        virtual void load_batch(Batch<Dtype>* batch);
        void transform_data_label(Blob<Dtype>* data_blob,Blob<Dtype>* label_blob, const cv::Mat& img,
                                  float img_width_scale, float img_height_scale);
        void generate_hm(const vector<cv::Point>& hm_pts,Blob<Dtype>* transformed_blob);
        int Rand(int n);
        cv::Mat RotateImage(cv::Mat& src, cv::Mat& dst, float angle);
        float Uniform(const float min, const float max);

        vector<vector<string> > lines_;
        Blob<Dtype> trans_data_tmp_,trans_label_tmp_;
        int lines_id_;
    };



}  // namespace caffe
#endif //CAFFE_HEATMAP_DATA_LAYER_HPP
