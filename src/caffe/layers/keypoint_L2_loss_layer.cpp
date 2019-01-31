#include <vector>

#include "caffe/layers/keypoint_L2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void KeypointL2LossLayer<Dtype>::Reshape(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        LossLayer<Dtype>::Reshape(bottom, top);
//        CHECK_EQ(bottom[0]->count(2), bottom[1]->count(2)) //向量维度相等
//                    << "Inputs must have the same dimension.";
        CHECK_EQ(bottom[0]->num(),bottom[1]->num())<<"Inputs must have the same num.";
        CHECK_EQ(bottom[0]->channels(),bottom[1]->channels())<<"Inputs must have the same channels.";
        CHECK_EQ(bottom[0]->height(),bottom[1]->height())<<"Inputs must have the same height.";
        CHECK_EQ(bottom[0]->width(),bottom[1]->width())<<"Inputs must have the same width.";
        diff_.ReshapeLike(*bottom[0]);
    }

    template<typename Dtype>
    void KeypointL2LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
        int count = bottom[0]->count();
        int n = bottom[0]->num();
        int c = bottom[0]->channels();
        int dim = bottom[0]->count(2);
        caffe_sub(
                count,
                bottom[0]->cpu_data(),
                bottom[1]->cpu_data(),
                diff_.mutable_cpu_data());
        Dtype* diff_data=diff_.mutable_cpu_data();
        const Dtype* label_data=bottom[1]->cpu_data();
        //自定义
//        Dtype loss=0;
//        for(int j=0;j<c;++j)
//        {
//            Dtype joint_loss=0;
//            for(int i=0;i<n;++i)
//            {
//                if(label_data[(i*c+j)*dim] < 0)
//                {
//                    for(int k=0;k<dim;++k)
//                    {
//                        diff_data[(i*c+j)*dim+k]=0;
//                    }
//                }
//                else
//                {
//                    for(int k=0;k<dim;++k)
//                    {
//                        joint_loss+=diff_data[(i*c+j)*dim+k]*diff_data[(i*c+j)*dim+k];
//                    }
//                }
//            }
//            loss+=joint_loss/n/dim;
//        }
//        loss/=c*Dtype(2);
        for(int i=0;i<n;++i)
        {
            for(int j=0;j<c;++j)
            {
                if(label_data[(i*c+j)*dim] < 0)
                {
                    for(int k=0;k<dim;++k)
                    {
                        diff_data[(i*c+j)*dim+k]=0;
                    }
                }
            }
        }
        Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
        Dtype loss = dot / bottom[0]->num() / Dtype(2);
        top[0]->mutable_cpu_data()[0] = loss;
//        for(int j=0;j<64;++j)
//        {
//            std::cout<<"前向   diff_.count():"<<diff_.count()<<std::endl;
//            std::cout<<"前向   diff_.cpu_data()[64*64*12*5+j]:"<<diff_.cpu_data()[64*64*12*5+j]<<std::endl;
//            std::cout<<"前向   diff_.cpu_data()[64*64*12*5+j]:"<<bottom[0]->cpu_data()[64*64*12*5+j]<<std::endl;
//            std::cout<<"前向   diff_.cpu_data()[64*64*12*5+j]:"<<bottom[1]->cpu_data()[64*64*12*5+j]<<std::endl;
//        }
    }

    template<typename Dtype>
    void KeypointL2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                                  const vector<bool> &propagate_down,
                                                  const vector<Blob<Dtype> *> &bottom) {
        //自定义
//        int n = bottom[0]->num();
//        int c = bottom[0]->channels();
//        int dim = bottom[0]->count(2);
//        Dtype* diff_data=diff_.mutable_cpu_data();
//        for (int id = 0; id < 2; ++id)
//        {
//            if (propagate_down[id])
//            {
//                const Dtype sign = (id == 0) ? 1 : -1;
//                const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[id]->count();
//                Dtype *bottom_diff=bottom[id]->mutable_cpu_data();
//                for(int j=0;j<c;++j)
//                {
//                    for(int i=0;i<n;++i)
//                    {
//                        for(int k=0;k<dim;++k)
//                        {
//                            bottom_diff[(i*c+j)*dim+k]=diff_data[(i*c+j)*dim+k]*alpha;
//                        }
//                    }
//                }
//            }
//        }
        for (int i = 0; i < 2; ++i)
        {
            if (propagate_down[i])
            {
                const Dtype sign = (i == 0) ? 1 : -1;
                const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
                caffe_cpu_axpby(
                        bottom[i]->count(),              // count
                        alpha,                              // alpha
                        diff_.cpu_data(),                   // a
                        Dtype(0),                           // beta
                        bottom[i]->mutable_cpu_diff());  // b
            }
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(KeypointL2LossLayer);
#endif

    INSTANTIATE_CLASS(KeypointL2LossLayer);

    REGISTER_LAYER_CLASS(KeypointL2Loss);

}  // namespace caffe
