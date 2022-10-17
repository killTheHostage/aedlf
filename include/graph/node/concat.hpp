#pragma once
#include "common/base.hpp"
#include <cstddef>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class ConcatNode : public BaseNode<MType> {
            public:
                using ul_pos = const std::pair<int, int>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using kernel_shape = std::vector<unsigned>;
                using matrix_dim = std::vector<int>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                ConcatNode(std::string node_name, matrix_dim m_dim, int concat_dim) : BaseNode<MType> {node_name, m_dim}, concat_dim_(concat_dim) {};
                ConcatNode(std::string node_name, const Matrix<MType>& m, int concat_dim) : BaseNode<MType> {node_name, m}, concat_dim_(concat_dim) {};
                ConcatNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, int concat_dim) : BaseNode<MType> {node_name, data, m_dim}, concat_dim_(concat_dim) {};
                ConcatNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent, int concat_dim) : BaseNode<MType> {node_name, data, m_dim, parent}, concat_dim_(concat_dim) {};
                ConcatNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents, int concat_dim) : BaseNode<MType> {node_name, data, m_dim, parents}, concat_dim_(concat_dim) {};
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
                void backward(node_ptr output_node) override;
            protected:
                matrix_tools::MakeMatrix<MType> mm;
                int concat_dim_;
        };

        template <typename MType>
        void ConcatNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            Matrix<MType>::data = BaseNode<MType>::get_parent(0)->get_data();
            for(size_t i {1}; i < parents_len; ++i) {
                Matrix<MType>::data.concat(BaseNode<MType>::get_parent(0)->get_dim(), concat_dim_);
            }
        }

        template <typename MType>
        void ConcatNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            int parent_i {0};
            while(parent_node != BaseNode<MType>::get_parent(parent_i) && parent_i != parents_len) {
                ++parent_i;
            }
            assert(parent_i < parents_len);
            m = BaseNode<MType>::get_parent(parent_i)->get_data().slice(1, {parent_i, parent_i+1});
        }

        template <typename MType>
        void ConcatNode<MType>::backward(node_ptr output_node) {
            compute_jacobi(BaseNode<MType>::jacobi, output_node);
            BaseNode<MType>::wait_backward = false;
        }
    }
}