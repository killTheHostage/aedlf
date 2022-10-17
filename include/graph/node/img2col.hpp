#pragma once
#include "common/base.hpp"


namespace aedlf {
    namespace graph {
        template <typename MType>
        class Img2colNode : public BaseNode<MType> {
            public:
                using ul_pos = const std::pair<int, int>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_dim = std::vector<int>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                using kernel_shape = std::vector<unsigned>;
                Img2colNode(std::string node_name, matrix_dim m_dim, kernel_shape kernel_size, int stride) : BaseNode<MType> {node_name, m_dim}, stride_(stride), kernel_size_(kernel_size) {};
                Img2colNode(std::string node_name, const Matrix<MType>& m, kernel_shape kernel_size, int stride) : BaseNode<MType> {node_name, m}, stride_(stride), kernel_size_(kernel_size) {};
                Img2colNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, kernel_shape kernel_size, int stride) : BaseNode<MType> {node_name, data, m_dim}, stride_(stride), kernel_size_(kernel_size) {};
                Img2colNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent, kernel_shape kernel_size, int stride) : BaseNode<MType> {node_name, data, m_dim, parent}, stride_(stride), kernel_size_(kernel_size) {};
                Img2colNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents, kernel_shape kernel_size, int stride) : BaseNode<MType> {node_name, data, m_dim, parents}, stride_(stride), kernel_size_(kernel_size) {};
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
                void backward(node_ptr output_node) override;
            protected:
                matrix_tools::MakeMatrix<MType> mm;
                int stride_;
                kernel_shape kernel_size_;
        };

        template <typename MType>
        void Img2colNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 1);
            mm.img2col(BaseNode<MType>::get_parent(0)->get_data(), BaseNode<MType>::data, kernel_size_, stride_);
        }

        template <typename MType>
        void Img2colNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            assert(parent_node == BaseNode<MType>::get_parent(0));
            mm.col2img(BaseNode<MType>::data, m, kernel_size_, stride_, parent_node->get_data_dim());
        }

        template <typename MType>
        void Img2colNode<MType>::backward(node_ptr output_node) {
            compute_jacobi(BaseNode<MType>::jacobi, output_node);
            BaseNode<MType>::wait_backward = false;
        }
    }
}