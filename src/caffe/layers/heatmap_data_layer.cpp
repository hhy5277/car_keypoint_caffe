#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/heatmap_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template<typename Dtype>
    HeatmapDataLayer<Dtype>::~HeatmapDataLayer<Dtype>() {
        this->StopInternalThread();
    }

    template<typename Dtype>
    void HeatmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
        const int new_height = this->layer_param_.heatmap_data_param().new_height();
        const int new_width = this->layer_param_.heatmap_data_param().new_width();
        const int crop_size = this->layer_param_.heatmap_data_param().crop_size();
        const int points_num = this->layer_param_.heatmap_data_param().coordinate_num();
        const bool is_color = this->layer_param_.heatmap_data_param().is_color();
        string root_folder = this->layer_param_.heatmap_data_param().root_folder();
        const int label_size = this->layer_param_.heatmap_data_param().label_size();
        //original code
//  CHECK((new_height == 0 && new_width == 0) ||
//      (new_height > 0 && new_width > 0)) << "Current implementation requires "
//      "new_height and new_width to be set at the same time.";
        //my code
        CHECK(new_height > 0 && new_width > 0) << "Both new_width and new_height must be greater than 0.";
        // Read the file with filenames and labels
        CHECK(new_height >= crop_size && new_width >= crop_size)
        << "Both new_width and new_height must greater than crop_size.";
        const string &source = this->layer_param_.heatmap_data_param().source();
        LOG(INFO) << "Opening file " << source;
        std::ifstream infile(source.c_str());
        string line;
        vector<string> line_split;
        while (std::getline(infile, line))
        {
//    pos = line.find_last_of(' ');
//    label = atoi(line.substr(pos + 1).c_str());
//    lines_.push_back(std::make_pair(line.substr(0, pos), label));

            boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
            CHECK_EQ((int) line_split.size() - 1, points_num*2) << "points num is not same with source.";
            lines_.push_back(line_split);
        }

        CHECK(!lines_.empty()) << "File is empty";

        if (this->layer_param_.heatmap_data_param().shuffle())
        {
            // randomly shuffle data
            LOG(INFO) << "Shuffling data";
            const unsigned int prefetch_rng_seed = caffe_rng_rand();
            prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
            ShuffleImages();
        } else
        {
            if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
                this->layer_param_.heatmap_data_param().rand_skip() == 0)
            {
                LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
            }
        }
        LOG(INFO) << "A total of " << lines_.size() << " images.";

        lines_id_ = 0;
        // Check if we would need to randomly skip a few data points
        if (this->layer_param_.heatmap_data_param().rand_skip())
        {
            unsigned int skip = caffe_rng_rand() %
                                this->layer_param_.heatmap_data_param().rand_skip();
            LOG(INFO) << "Skipping first " << skip << " data points.";
            CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
            lines_id_ = skip;
        }
        // Read an image, and use it to initialize the top blob.
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
                                          new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
        // Use data_transformer to infer the expected blob shape from a cv_image.
//  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
        vector<int> top_shape(4);
        top_shape[0] = 1;
        top_shape[1] = cv_img.channels();
        if (crop_size)
        {
            top_shape[2] = crop_size;
            top_shape[3] = crop_size;
        } else
        {
            top_shape[2] = new_height;
            top_shape[3] = new_width;
        }
        trans_data_tmp_.Reshape(top_shape);
        // Reshape prefetch_data and top[0] according to the batch_size.
        const int batch_size = this->layer_param_.heatmap_data_param().batch_size();
        CHECK_GT(batch_size, 0) << "Positive batch size required";
        top_shape[0] = batch_size;
        top[0]->Reshape(top_shape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
        {
            this->prefetch_[i]->data_.Reshape(top_shape);
        }
        LOG(INFO) << "output data size: " << top[0]->num() << ","
                  << top[0]->channels() << "," << top[0]->height() << ","
                  << top[0]->width();
        // label
//  vector<int> label_shape(1, batch_size);
        vector<int> label_shape(4);
        label_shape[0] = 1;
        label_shape[1] = points_num;
        label_shape[2] = label_size;
        label_shape[3] = label_size;
        trans_label_tmp_.Reshape(label_shape);
        label_shape[0] = batch_size;
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
        {
            this->prefetch_[i]->label_.Reshape(label_shape);
        }
    }

    template<typename Dtype>
    void HeatmapDataLayer<Dtype>::ShuffleImages() {
        caffe::rng_t *prefetch_rng =
                static_cast<caffe::rng_t *>(prefetch_rng_->generator());
        shuffle(lines_.begin(), lines_.end(), prefetch_rng);
    }

// This function is called on prefetch thread
    template<typename Dtype>
    void HeatmapDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK(batch->data_.count());
        CHECK(trans_data_tmp_.count());
        HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();
        const int batch_size = heatmap_data_param.batch_size();
        const int new_height = heatmap_data_param.new_height();
        const int new_width = heatmap_data_param.new_width();
        const int crop_size = heatmap_data_param.crop_size();
//  const int points_num=heatmap_data_param.coordinate_num();
//  const int label_size=heatmap_data_param.label_size();
        const bool is_color = heatmap_data_param.is_color();
        string root_folder = heatmap_data_param.root_folder();

        // Reshape according to the first image of each batch
        // on single input batches allows for inputs of varying dimension.
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0], new_height, new_width, is_color);
//  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
//  // Use data_transformer to infer the expected blob shape from a cv_img.
//  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
//  this->transformed_data_.Reshape(top_shape);
//  // Reshape batch according to the batch_size.

        vector<int> top_shape(4);
        top_shape[0] = 1;
        top_shape[1] = cv_img.channels();
        if (crop_size)
        {
            top_shape[2] = crop_size;
            top_shape[3] = crop_size;
        } else
        {
            top_shape[2] = new_height;
            top_shape[3] = new_width;
        }

        trans_data_tmp_.Reshape(top_shape);
        top_shape[0] = batch_size;
        batch->data_.Reshape(top_shape);


        Dtype *prefetch_data = batch->data_.mutable_cpu_data();
        Dtype *prefetch_label = batch->label_.mutable_cpu_data();//自加

        // datum scales
        const int lines_size = lines_.size();
        for (int item_id = 0; item_id < batch_size; ++item_id)
        {
            // get a blob
            timer.Start();
            CHECK_GT(lines_size, lines_id_);
//    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
//        new_height, new_width, is_color);
            cv::Mat img;
            int cv_read_flag = is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
            string filename = root_folder + lines_[lines_id_][0];
            cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
            if (!cv_img_origin.data)
            {
                LOG(ERROR) << "Could not open or find file " << filename;
            }
            if (new_height > 0 && new_width > 0)
            {
                cv::resize(cv_img_origin, img, cv::Size(new_width, new_height));
            } else
            {
                img = cv_img_origin;
            }
            float img_width_scale = (float) img.cols / cv_img_origin.cols;
            float img_height_scale = (float) img.rows / cv_img_origin.rows;

            CHECK(img.data) << "Could not load " << lines_[lines_id_][0];
            read_time += timer.MicroSeconds();
            timer.Start();
            // Apply transformations (mirror, crop...) to the image
            int data_offset = batch->data_.offset(item_id);
            int label_offset = batch->label_.offset(item_id);
            //original code
//    this->transformed_data_.set_cpu_data(prefetch_data + offset);
//    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
            //my code
            trans_data_tmp_.set_cpu_data(prefetch_data + data_offset);
            trans_label_tmp_.set_cpu_data(prefetch_label + label_offset);
            transform_data_label(&trans_data_tmp_, &trans_label_tmp_, img, img_width_scale, img_height_scale);

            trans_time += timer.MicroSeconds();

//    prefetch_label[item_id] = lines_[lines_id_][1];
            // go to the next iter
            lines_id_++;
            if (lines_id_ >= lines_size)
            {
                // We have reached the end. Restart from the first.
                DLOG(INFO) << "Restarting data prefetching from start.";
                lines_id_ = 0;
                if (this->layer_param_.image_data_param().shuffle())
                {
                    ShuffleImages();
                }
            }
        }
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    template<typename Dtype>
    void
    HeatmapDataLayer<Dtype>::transform_data_label(Blob<Dtype> *data_blob, Blob<Dtype> *label_blob, const cv::Mat &img,
                                                  float img_width_scale, float img_height_scale) {
        const bool has_mean_file = this->layer_param_.heatmap_data_param().has_mean_file();
        const string mean_file = this->layer_param_.heatmap_data_param().mean_file();
        const bool do_rotation = this->layer_param_.heatmap_data_param().rotation() && Rand(2);
        const float max_angle = this->layer_param_.heatmap_data_param().max_angle();
        const int mirror_pairs_size = this->layer_param_.heatmap_data_param().mirror_pairs_size();
        const int points_num = this->layer_param_.heatmap_data_param().coordinate_num();

        std::vector<string> mirror_pairs;
        for (int i = 0; i < mirror_pairs_size; ++i)
        {
            mirror_pairs.push_back(this->layer_param_.heatmap_data_param().mirror_pairs(i));
        }

        const int crop_size = this->layer_param_.heatmap_data_param().crop_size();
        const int img_channels = img.channels();
        const int img_height = img.rows;
        const int img_width = img.cols;

        // Check dimensions.
        const int channels = data_blob->channels();
        const int height = data_blob->height();
        const int width = data_blob->width();
        const int num = data_blob->num();

        CHECK_EQ(channels, img_channels);
        CHECK_LE(height, img_height);
        CHECK_LE(width, img_width);
        CHECK_GE(num, 1);

        CHECK(img.depth() == CV_8U) << "Image data type must be unsigned byte";


        Blob<Dtype> data_mean;
        BlobProto blob_proto;
        if (has_mean_file)
        {
            ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
            data_mean.FromProto(blob_proto);
        }

        const Dtype scale = this->layer_param_.heatmap_data_param().scale();
        const bool do_mirror = this->layer_param_.heatmap_data_param().mirror() && Rand(2);
//    const bool has_mean_file = this->layer_param_.heatmap_data_param().has_mean_file();

        vector<Dtype> mean_values;
        for (int c = 0; c < this->layer_param_.heatmap_data_param().mean_value_size(); ++c)
        {
            mean_values.push_back(this->layer_param_.heatmap_data_param().mean_value(c));
        }
        const bool has_mean_values = mean_values.size() > 0;

        CHECK_GT(img_channels, 0);
        CHECK_GE(img_height, crop_size);
        CHECK_GE(img_width, crop_size);

        Dtype *mean = NULL;
        if (has_mean_file)
        {
            CHECK_EQ(img_channels, data_mean.channels());
            CHECK_EQ(img_height, data_mean.height());
            CHECK_EQ(img_width, data_mean.width());
            mean = data_mean.mutable_cpu_data();
        }
        if (has_mean_values)
        {
            CHECK(mean_values.size() == 1 || mean_values.size() == img_channels)
            << "Specify either 1 mean_value or as many as channels: " << img_channels;
            if (img_channels > 1 && mean_values.size() == 1)
            {
                // Replicate the mean_value for simplicity
                for (int c = 1; c < img_channels; ++c)
                {
                    mean_values.push_back(mean_values[0]);
                }
            }
        }

        ///随机裁剪
        int h_off = 0;
        int w_off = 0;
        cv::Mat cv_cropped_img = img;
        if (crop_size)
        {
            CHECK_EQ(crop_size, height);
            CHECK_EQ(crop_size, width);
            // We only do random crop when we do training.
            if (this->phase_ == TRAIN)
            {
                h_off = Rand(img_height - crop_size + 1);
                w_off = Rand(img_width - crop_size + 1);
            } else
            {
                h_off = (img_height - crop_size) / 2;
                w_off = (img_width - crop_size) / 2;
            }
            cv::Rect roi(w_off, h_off, crop_size, crop_size);
            cv_cropped_img = img(roi);
        } else
        {
            CHECK_EQ(img_height, height);
            CHECK_EQ(img_width, width);
        }

        CHECK(cv_cropped_img.data);
        cv::Mat cv_rotated_img = cv_cropped_img;
        float angle = 0;
        if (do_rotation)
        {
            angle = Uniform(-max_angle, max_angle);
//        angle =90;
            RotateImage(cv_cropped_img, cv_rotated_img, angle);
        }
        CHECK(cv_rotated_img.data);
        //data处理
        Dtype *transform_data = trans_data_tmp_.mutable_cpu_data();
        int top_index;
        for (int h = 0; h < height; ++h)
        {
            const uchar *ptr = cv_rotated_img.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < img_channels; ++c)
                {
                    if (do_mirror)
                    {
                        top_index = (c * height + h) * width + (width - 1 - w);//Blob进行镜像
                    } else
                    {
                        top_index = (c * height + h) * width + w;
                    }
                    // int top_index = (c * height + h) * width + w;
                    Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                    if (has_mean_file)
                    {
                        int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
                        transform_data[top_index] = (pixel - mean[mean_index]) * scale;
                    }
                    else
                    {
                        if (has_mean_values)
                        {
                            transform_data[top_index] = (pixel - mean_values[c]) * scale;
                        } else
                        {
                            transform_data[top_index] = pixel * scale;
                        }
                    }
                }
            }
        }

        //label处理
        //随机剪裁
        int coords_size = (int) lines_[lines_id_].size() - 1;
        CHECK_EQ(coords_size, points_num*2) << "keypoints must has has points_nums points.";
        vector<cv::Point> points(points_num);
        for (int i = 0; i < points_num; ++i)
        {
            vector<string> numstr_split;
            numstr_split.push_back(lines_[lines_id_][2*i + 1]);
            numstr_split.push_back(lines_[lines_id_][2*i + 2]);
            CHECK_EQ(numstr_split.size(), 2) << "coordinate param has two numbers.";
            int x_ori = std::atoi(numstr_split[0].c_str());
            int y_ori = std::atoi(numstr_split[1].c_str());
            int x = x_ori>0 ? (int)roundf(x_ori * img_width_scale) - w_off : -1;//-1变成0
            int y = y_ori>0 ? (int)roundf(y_ori * img_height_scale) - h_off : -1;
//            if(numstr_split[0]=="-1")
//            {
//                std::cout<<"x:"<<x<<std::endl;
//                std::cout<<"img_width_scale:"<<img_width_scale<<std::endl;
//                std::cout<<"w_off:"<<w_off<<std::endl;
//            }
//        x = x>=0 ? x : -1;
//        y = y>=0 ? y : -1;
            points[i].x = x;
            points[i].y = y;
        }
        //旋转 getRotationMatrix2D 与实际方向相反
        float arc_angle = angle * M_PI / -180;
        if (do_rotation)
        {
            float center_x = width / 2.0f;
            float center_y = height / 2.0f;
            for (int i = 0; i < points_num; ++i)
            {
                float r_x = points[i].x - center_x;
                float r_y = points[i].y - center_y;
                int x_rotated = (int)roundf(center_x + r_x * std::cos(arc_angle) - r_y * std::sin(arc_angle));
                int y_rotated = (int)roundf(center_y + r_y * std::cos(arc_angle) + r_x * std::sin(arc_angle));
                points[i].x = x_rotated;
                points[i].y = y_rotated;
//            if(x_rotated>=0 && x_rotated<width && y_rotated>=0 && y_rotated<height)
//            {
//                points[i].x=x_rotated;
//                points[i].y=y_rotated;
//            }
//            else
//            {
//                points[i].x=-1;
//                points[i].y=-1;
//            }
            }
        }
        //处理crop和rotation使得关键点被剔除
        for (int i = 0; i < points_num; ++i)
        {
            int ptx = points[i].x;
            int pty = points[i].y;
            if (ptx >= 0 && ptx < width && pty >= 0 && pty < height)
            {
//                if(points[i].x==0)
//                {
//                    std::cout<<"处理crop和rotation使得关键点被剔除"<<points[i].x<<std::endl;
//                }
                points[i].x = ptx;
                points[i].y = pty;
            } else
            {
                points[i].x = -1;
                points[i].y = -1;

            }
        }
        //镜像
        if (do_mirror)
        {
            for (int i = 0; i < points_num; ++i)
            {
                if (points[i].x != -1 && points[i].y != -1)
                {
                    points[i].x = width - 1 - points[i].x;
                }
            }
            //关键点重新排序
            std::vector<std::string> kpt;
            for (int i = 0; i < mirror_pairs_size; ++i)
            {
                std::string &pair = mirror_pairs[i];
                kpt.clear();
                boost::split(kpt, pair, boost::is_any_of(" ,"), boost::token_compress_on);
                CHECK_EQ((int) kpt.size(), 2) << "mirror_pair must has two element.";
                int one_idx = std::atoi(kpt[0].c_str());
                int ano_idx = std::atoi(kpt[1].c_str());
                int x_tmp = points[one_idx].x;
                int y_tmp = points[one_idx].y;
                points[one_idx].x = points[ano_idx].x;
                points[one_idx].y = points[ano_idx].y;
                points[ano_idx].x = x_tmp;
                points[ano_idx].y = y_tmp;
            }
        }
        ///生成heatmap
        const int hm_size = this->layer_param_.heatmap_data_param().label_size();
        float hm_width_scale = (float) hm_size / width;
        float hm_height_scale = (float) hm_size / height;
        vector<cv::Point> hm_pts(points.begin(), points.end());
        for (int i = 0; i < points_num; ++i)
        {
            if (hm_pts[i].x < 0 || hm_pts[i].y < 0)
            {
//                if(hm_pts[i].x==-1)
//                    std::cout<<"hm_pts[i].x"<<hm_pts[i].x<<std::endl;
                continue;
            }
//            std::cout<<"hm_pts[i]"<<hm_pts[i].x<<" "<<hm_pts[i].y<<std::endl;
            hm_pts[i].x = (int)roundf(hm_width_scale * hm_pts[i].x);
            hm_pts[i].y = (int)roundf(hm_height_scale * hm_pts[i].y);
//            std::cout<<"hm_pts[i]"<<hm_pts[i].x<<" "<<hm_pts[i].y<<std::endl<<std::endl;
        }
        generate_hm(hm_pts, label_blob);
    }

    template<typename Dtype>
    void HeatmapDataLayer<Dtype>::generate_hm(const vector<cv::Point> &hm_pts, Blob<Dtype> *label_blob) {
        CHECK(hm_pts.size() == label_blob->channels()) << "heatmap points are same with channels of label_blob";
        CHECK(label_blob->shape().size() == 4) << "label Blob must has 4 dimension.";
        CHECK(label_blob->num() == 1) << "label Blob .num() must be 1.";
        const int label_height = label_blob->height();
        const int label_width = label_blob->width();
        const int label_num_channels = label_blob->channels();
        const int label_channel_size = label_height * label_width;
//    const int label_img_size = label_channel_size * label_num_channels / 2;
        caffe_set(label_blob->count(),Dtype(0),label_blob->mutable_cpu_data());
        Dtype *label_ptr = label_blob->mutable_cpu_data();

//    cv::Mat dataMatrix = cv::Mat::zeros(label_height, label_width, CV_32FC1);
//    float label_resize_fact = (float) label_height / (float) outsize;
        float sigma = 2;

        for (int idx_ch = 0; idx_ch < label_num_channels; idx_ch++)
        {
            if (hm_pts[idx_ch].x < 0 || hm_pts[idx_ch].y < 0)
            {
                for (int i = 0; i < label_height; i++)
                {
                    for (int j = 0; j < label_width; j++)
                    {
                        // 计算索引
                        int label_idx = idx_ch * label_channel_size + i * label_width + j;
                        label_ptr[label_idx] = Dtype(-1);
//                        std::cout<<"label_ptr[label_idx]:"<<label_ptr[label_idx]<<" "<<idx_ch<<" "<<i<<" "<<j<<std::endl;
                    }
                }
            }
            else
            {
                for (int i = 0; i < label_height; i++)
                {
                    for (int j = 0; j < label_width; j++)
                    {
                        // 计算索引
                        int label_idx = idx_ch * label_channel_size + i * label_width + j;
                        float gaussian = (float)exp(-0.5 * (pow(i - hm_pts[idx_ch].y, 2) + pow(j - hm_pts[idx_ch].x, 2)) * pow(1 / sigma, 2));
                        //for debug
//                        if(i== hm_pts[idx_ch].y && j==hm_pts[idx_ch].x)
//                        {
//                            std::cout<<"gaussian:"<<gaussian<<" "<<j<<" "<<i<<std::endl;
//                        }
                        if(gaussian>0.0001)
                        {
                            label_ptr[label_idx] = gaussian;
                        }
                    }
                }
            }
        }
    }

    template<typename Dtype>
    int HeatmapDataLayer<Dtype>::Rand(int n) {
        shared_ptr<Caffe::RNG> rng_;
        const unsigned int rng_seed = caffe_rng_rand();
        rng_.reset(new Caffe::RNG(rng_seed));
        CHECK_GT(n, 0);
        caffe::rng_t *rng = static_cast<caffe::rng_t *>(rng_->generator());
        return ((*rng)() % n);
    }

    template<typename Dtype>
    cv::Mat HeatmapDataLayer<Dtype>::RotateImage(cv::Mat &src, cv::Mat &dst, float angle) {
        cv::Point center(src.cols / 2, src.rows / 2);
        double scale = 1;
        // Get the rotation matrix with the specifications above
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        // Rotate the warped image
        cv::warpAffine(src, dst, rot_mat, src.size());
        return rot_mat;
    }

    template<typename Dtype>
    float HeatmapDataLayer<Dtype>::Uniform(const float min, const float max) {
        float random = ((float) rand()) / (float) RAND_MAX;
        float diff = max - min;
        float r = random * diff;
        return min + r;
    }

    INSTANTIATE_CLASS(HeatmapDataLayer);

    REGISTER_LAYER_CLASS(HeatmapData);

}  // namespace caffe
#endif  // USE_OPENCV
