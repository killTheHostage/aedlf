#pragma once
#include "common/base.hpp"


namespace aedlf {
    namespace graph {
        template <typename MType>
        class PaddingNode : public BaseNode<MType> {
            public:
                using ul_pos = const std::pair<int, int>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using kernel_shape = std::vector<unsigned>;
                using matrix_dim = std::vector<int>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                PaddingNode(std::string node_name, matrix_dim m_dim, kernel_shape padding, MType padding_init) : BaseNode<MType> {node_name, m_dim}, padding_size_(padding), padding_init_(padding_init) {};
                PaddingNode(std::string node_name, const Matrix<MType>& m, kernel_shape padding, MType padding_init) : BaseNode<MType> {node_name, m}, padding_size_(padding), padding_init_(padding_init) {};
                PaddingNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, kernel_shape padding, MType padding_init) : BaseNode<MType> {node_name, data, m_dim}, padding_size_(padding), padding_init_(padding_init) {};
                PaddingNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent, kernel_shape padding, MType padding_init) : BaseNode<MType> {node_name, data, m_dim, parent}, padding_size_(padding), padding_init_(padding_init) {};
                PaddingNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents, kernel_shape padding, MType padding_init) : BaseNode<MType> {node_name, data, m_dim, parents}, padding_size_(padding), padding_init_(padding_init) {};
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
                void backward(node_ptr output_node) override;
            protected:
                matrix_tools::MakeMatrix<MType> mm;
                kernel_shape padding_size_;
                MType padding_init_;
        };

        template <typename MType>
        void PaddingNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 1);
            if(padding_size_[0] == 0 && padding_size_[1] == 0) {
                BaseNode<MType>::data = BaseNode<MType>::get_parent(0)->get_data();
                return;
            }
            mm.modify_dim(BaseNode<MType>::data.get_dim());
            mm.add_padding(BaseNode<MType>::get_parent(0), BaseNode<MType>::data, padding_size_, padding_init_);
        }

        template <typename MType>
        void PaddingNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            assert(parent_node == BaseNode<MType>::get_parent(0));
            mm.modify_dim(parent_node->get_data_dim());
            mm.sub_padding(BaseNode<MType>::data, m, padding_size_);
        }

        template <typename MType>
        void PaddingNode<MType>::backward(node_ptr output_node) {
            compute_jacobi(BaseNode<MType>::jacobi, output_node);
            BaseNode<MType>::wait_backward = false;
        }
    }
}