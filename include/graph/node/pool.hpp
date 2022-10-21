#pragma once
#include "common/base.hpp"
#include <map>
#include <thread>
#include <algorithm>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class MaxPool2dNode : public BaseNode<MType> {
            public:
                using ul_pos = const std::pair<unsigned long, unsigned long>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_dim = std::vector<unsigned long>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                using kernel_shape = std::vector<unsigned long>;
                using not_zero_index_c = std::vector<unsigned long>;
                MaxPool2dNode(std::string node_name, matrix_dim m_dim, kernel_shape kernel_size, unsigned long stride) : BaseNode<MType> {node_name, m_dim}, stride_(stride), kernel_size_(kernel_size) {};
                MaxPool2dNode(std::string node_name, const Matrix<MType>& m, kernel_shape kernel_size, unsigned long stride) : BaseNode<MType> {node_name, m}, stride_(stride), kernel_size_(kernel_size) {};
                MaxPool2dNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, kernel_shape kernel_size, unsigned long stride) : BaseNode<MType> {node_name, data, m_dim}, stride_(stride), kernel_size_(kernel_size) {};
                MaxPool2dNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent, kernel_shape kernel_size, unsigned long stride) : BaseNode<MType> {node_name, data, m_dim, parent}, stride_(stride), kernel_size_(kernel_size) {};
                MaxPool2dNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents, kernel_shape kernel_size, unsigned long stride) : BaseNode<MType> {node_name, data, m_dim, parents}, stride_(stride), kernel_size_(kernel_size) {};
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
                void backward(node_ptr output_node) override;
            protected:
                void pooling_core(Matrix<MType>& m, Matrix<MType>& fw, ul_pos m_channel_ul, ul_pos fw_channel_ul);
                void depooling_core(Matrix<MType>& m, ul_pos data_channel_ul, unsigned long nzic_index);
                int stride_;
                kernel_shape kernel_size_;
                not_zero_index_c nzic;
        };

        template <typename MType>
        void MaxPool2dNode<MType>::compute_forward() {
            nzic.clear();
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 1);
            matrix_dim fw_dim {BaseNode<MType>::get_parent(0)->get_dim()};
            fw_dim[2] = fw_dim[2] / kernel_size_[0];
            fw_dim[3] = fw_dim[3] / kernel_size_[1];
            Matrix<MType> fw {fw_dim, 0};
            Matrix<MType> m {BaseNode<MType>::get_parent(0)->get_data()};
            std::vector<std::thread> thread_c;
            for(unsigned long n {0}; n < fw_dim[0]; ++n) {
                ul_pos m_batch_ul {m.get_batch(n)};
                ul_pos fw_batch_ul {fw.get_batch(n)};
                for(unsigned long c {0}; c < fw_dim[1]; ++c) {
                    ul_pos m_channel_ul {m.get_channel(c, m_batch_ul)};
                    ul_pos fw_channel_ul {fw.get_channel(c, fw_batch_ul)};
                    thread_c.emplace_back(&MaxPool2dNode<MType>::pooling_core, this, std::ref(m), std::ref(fw), m_channel_ul, fw_channel_ul);
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
            BaseNode<MType>::data = fw;
        }

        template <typename MType>
        void MaxPool2dNode<MType>::pooling_core(Matrix<MType>& m, Matrix<MType>& fw, ul_pos m_channel_ul, ul_pos fw_channel_ul) {
            matrix_dim fw_dim {fw.get_dim()};
            matrix_data_p m_data {m.get_m_data()};
            matrix_data_p fw_data {fw.get_m_data()};
            for(unsigned long fw_h {0}; fw_h < fw_dim[2]; ++fw_h) {
                for(unsigned long fw_w {0}; fw_w < fw_dim[3]; ++fw_w) {
                    MType max {m_data->at(m_channel_ul.first + fw_h * stride_ * fw_dim[3] + fw_w * stride_)};
                    unsigned long max_i {m_channel_ul.first + fw_h * stride_ * fw_dim[3] + fw_w * stride_};
                    for(unsigned long k_h {0}; k_h < kernel_size_[0]; ++k_h) {
                        for(unsigned long k_w {1}; k_w < kernel_size_[1]; ++k_w) {
                            if(m_data->at(m_channel_ul.first + (fw_h * stride_ + k_h) * fw_dim[3] + fw_w * stride_ + k_w) > max) {
                                max = m_data->at(m_channel_ul.first + (fw_h * stride_ + k_h) * fw_dim[3] + fw_w * stride_ + k_w);
                                max_i = m_channel_ul.first + (fw_h * stride_ + k_h) * fw_dim[3] + fw_w * stride_ + k_w;
                            }
                        }
                    }
                    fw_data->at(fw_channel_ul.first + fw_h * fw_dim[2] + fw_w) = max;
                    nzic.push_back(max_i);
                }
            }
        }

        template <typename MType>
        void MaxPool2dNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            std::sort(nzic.begin(), nzic.end());
            assert(parent_node == BaseNode<MType>::get_parent(0));
            matrix_dim fw_dim {BaseNode<MType>::get_parent(0)->get_dim()};
            std::vector<std::thread> thread_c;
            Matrix<MType> fw {fw_dim, 0};
            unsigned long nzic_index;
            for(unsigned long n {0}; n < fw_dim[0]; ++n) {
                ul_pos data_batch_ul {BaseNode<MType>::data->get_batch(n)};
                for(unsigned long c {0}; c < fw_dim[1]; ++c) {
                    nzic_index = (n * fw_dim[1] + c) * (fw_dim[2] * fw_dim[3]);
                    ul_pos data_channel_ul {BaseNode<MType>::data->get_channel(c, data_batch_ul)};
                    thread_c.emplace_back(&MaxPool2dNode<MType>::depooling_core, this, std::ref(m), data_channel_ul, nzic_index);
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }

        template <typename MType>
        void MaxPool2dNode<MType>::depooling_core(Matrix<MType>& m, ul_pos data_channel_ul, unsigned long nzic_index) {
            matrix_dim data_dim {BaseNode<MType>::data.get_dim()};
            for(unsigned long m_h {0}; m_h < data_dim[2]; ++m_h) {
                for(unsigned long m_w {0}; m_w < data_dim[3]; ++m_w) {
                    m.set(nzic[nzic_index + m_h * data_dim[3] + m_w], BaseNode<MType>::data.get(data_channel_ul.first + m_h * data_dim[3] + m_w));
                }
            }
        }

        template <typename MType>
        void MaxPool2dNode<MType>::backward(node_ptr output_node) {
            compute_jacobi(BaseNode<MType>::jacobi, output_node);
            BaseNode<MType>::wait_backward = false;
        }
    }
}