#pragma once
#include "common/base.hpp"


namespace aedlf {
    namespace graph {
        template <typename MType>
        class DataNode : public BaseNode<MType> {
            public:
                using BaseNode<MType>::BaseNode;
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using matrix_dim = std::vector<unsigned long>;
                DataNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent);
                DataNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents);
                graph_nodes get_parents() override;
                node_ptr get_parent(size_t parent_id) override;
                size_t get_parents_len() override;
                void add_parent(node_ptr parent) override;
                void compute_forward() override {};
                void forward() override {};
                void backward(node_ptr children) override {};
                void clear_jacobi() override {};
            protected:
                bool require_grad {false};
        };

        template <typename MType>
        DataNode<MType>::DataNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent) {
            throw std::runtime_error("`DataNode` is not allowed to modify `parent`");
        }

        template <typename MType>
        DataNode<MType>::DataNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents) {
            throw std::runtime_error("`DataNode` is not allowed to modify `parent`");
        }

        template <typename MType>
        typename DataNode<MType>::graph_nodes DataNode<MType>::get_parents() {
            throw std::runtime_error("`DataNode` is not allowed to get `parents`");
        }

        template <typename MType>
        typename DataNode<MType>::node_ptr DataNode<MType>::get_parent(size_t parent_id) {
            throw std::runtime_error("`DataNode` is not allowed to get `parents` by `parent_id`");
        }

        template <typename MType>
        size_t DataNode<MType>::get_parents_len() {
            return 0;
        }

        template <typename MType>
        void DataNode<MType>::add_parent(node_ptr parent) {
            std::runtime_error("`DataNode` is not allow to add `parent`");
        }
    }
}