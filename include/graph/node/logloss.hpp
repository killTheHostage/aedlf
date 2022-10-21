#pragma once
#include "common/base.hpp"
#include "common/loss.hpp"
#include <cstddef>
#include <cmath>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class LogLossNode : public LossNode<MType> {
            public:
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using matrix_dim = std::vector<unsigned long>;
                LogLossNode(std::string node_name, matrix_dim m_dim, std::string reduction = "mean") : LossNode<MType> {node_name, m_dim}, reduction_(reduction) {};
                LogLossNode(std::string node_name, const Matrix<MType>& m, std::string reduction = "mean") : LossNode<MType> {node_name, m}, reduction_(reduction) {};
                LogLossNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, std::string reduction = "mean") : LossNode<MType> {node_name, data, m_dim}, reduction_{reduction} {};
                LogLossNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent, std::string reduction = "mean") : LossNode<MType> {node_name, data, m_dim, parent}, reduction_{reduction} {};
                LogLossNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents, std::string reduction = "mean") : LossNode<MType> {node_name, data, m_dim, parents}, reduction_(reduction) {};
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
                void backward(node_ptr output_node) override;
            protected:
                std::string reduction_;
        };

        template <typename MType>
        void LogLossNode<MType>::compute_forward() {
            // 规定parent(1)为label label的值应当为0或1，parent(0)为predict
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 2);
            assert(BaseNode<MType>::get_parent(0)->get_data_dim() == BaseNode<MType>::get_parent(1)->get_data_dim());
            matrix_dim data_dim {BaseNode<MType>::get_parent(0)->get_data_dim()};
            assert(data_dim[3] == 1 && data_dim[2] == 1 && data_dim[1] == 1);
            Matrix<MType> label_data {BaseNode<MType>::get_parent(1)->get_data()};
            Matrix<MType> pred_data {BaseNode<MType>::get_parent(0)->get_data()};
            matrix_data_p label_p {label_data.get_m_data()};
            matrix_data_p pred_p {pred_data.get_m_data()};
            Matrix<MType> loss_value {{1,1,1,1}, 0};
            MType loss_sum {0};
            if(typeid(label_p->at(0)) == typeid(float) || typeid(label_p->at(0)) == typeid(double) || typeid(label_p->at(0)) == typeid(long double)) {
                for(size_t i {0}; i < label_p->size(); ++i) {
                    if(std::fabs(label_p->at(i) - 1) < 1e-4) {
                        MType loss_piece = std::log(pred_p->at(i)) * -1.0;
                        loss_sum += loss_piece;
                    }
                    else {
                        MType loss_piece = std::log(1.0 - pred_p->at(i)) * -1.0;
                        loss_sum += loss_piece;
                    }
                }
            }
            else {
                for(size_t i {0}; i < label_p->size(); ++i) {
                    if(label_p->at(i) == 1) {
                        loss_sum += std::log(pred_p->at(i)) * -1.0;
                    }
                    else {
                        loss_sum += std::log(1.0 - pred_p->at(i)) * -1.0;
                    }
                }
            }
            if(reduction_ == "mean") {
                loss_value.set(0, loss_sum / label_p->size());
            }
            BaseNode<MType>::data = loss_value;
        }

        template <typename MType>
        void LogLossNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 2);
            assert(BaseNode<MType>::get_parent(0)->get_data_dim() == BaseNode<MType>::get_parent(1)->get_data_dim());
            matrix_dim data_dim {BaseNode<MType>::get_parent(0)->get_data_dim()};
            assert(data_dim[3] == 1 && data_dim[2] == 1 && data_dim[1] == 1);
            m.resize(BaseNode<MType>::get_parent(0)->get_data_dim(), 0);
            Matrix<MType> label_data {BaseNode<MType>::get_parent(1)->get_data()};
            Matrix<MType> pred_data {BaseNode<MType>::get_parent(0)->get_data()};
            matrix_data_p label_p {label_data.get_m_data()};
            matrix_data_p pred_p {pred_data.get_m_data()};
            if(typeid(label_p->at(0)) == typeid(float) || typeid(label_p->at(0)) == typeid(double) || typeid(label_p->at(0)) == typeid(long double)) {
                for(size_t i {0}; i < label_p->size(); ++i) {
                    if(std::fabs(label_p->at(i) - 1) < 1e-4) {
                        m.set(i, -1 / pred_p->at(i));
                    }
                    else {
                        m.set(i, 1 / (1 - pred_p->at(i)));
                    }
                }
            }
            else {
                for(size_t i {0}; i < label_p->size(); ++i) {
                    if(label_p->at(i) == 1) {
                        m.set(i, -1 / pred_p->at(i));
                    }
                    else {
                        m.set(i, 1 / (1 - pred_p->at(i)));
                    }
                }
            }
        }

        template <typename MType>
        void LogLossNode<MType>::backward(node_ptr output_node) {
            compute_jacobi(BaseNode<MType>::jacobi, output_node);
            BaseNode<MType>::wait_backward = false;
        }
    }
}