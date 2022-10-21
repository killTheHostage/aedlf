#pragma once
#include "common/base.hpp"
#include <cstddef>
#include <cmath>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class SigmoidNode : public BaseNode<MType> {
            public:
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using matrix_dim = std::vector<unsigned long>;
                using BaseNode<MType>::BaseNode;
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
        };

        template <typename MType>
        void SigmoidNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 1);
            Matrix<MType> input_matrix {BaseNode<MType>::get_parent(0)->get_data()};
            matrix_data_p input_matrix_p {input_matrix.get_m_data()};
            for(size_t i {0}; i < input_matrix_p->size(); ++i) {
                input_matrix_p->at(i) = 1 / (1 + std::exp(-1 * input_matrix_p->at(i)));
            }
            BaseNode<MType>::data = input_matrix;
        }

        template <typename MType>
        void SigmoidNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 1 && parent_node == BaseNode<MType>::get_parent(0));
            matrix_dim data_dim {parent_node->get_data_dim()};
            Matrix<MType> parent_data {parent_node->get_data()};
            matrix_data_p data_p {parent_data.get_m_data()};
            m.resize(data_dim, 0);
            for(size_t i {0}; i < data_p->size(); ++i) {
                m.set(i, (data_p->at(i) * (1 - data_p->at(i))));
            }
        }
    }
}