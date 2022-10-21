#pragma once
#include "base.hpp"
#include <stdexcept>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class LossNode : public BaseNode<MType> {
            public:
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using matrix_dim = std::vector<unsigned long>;
                using BaseNode<MType>::BaseNode;
                graph_nodes get_childrens() override;
                node_ptr get_children(size_t children_id) override;
                size_t get_childrens_len() override;
                void add_children(node_ptr children) override;
                void backward(node_ptr output_node) override = 0;
        };

        template <typename MType>
        typename LossNode<MType>::graph_nodes LossNode<MType>::get_childrens() {
            throw std::runtime_error("`LossNode` is not allowed to get `childrens`");
        }

        template <typename MType>
        typename LossNode<MType>::node_ptr LossNode<MType>::get_children(size_t children_id) {
            throw std::runtime_error("`LossNode` is not allowed to get `childrens` by `children_id`");
        }

        template <typename MType>
        size_t LossNode<MType>::get_childrens_len() {
            return 0;
        }

        template <typename MType>
        void LossNode<MType>::add_children(node_ptr children) {
            throw std::runtime_error("`LossNode` is not allowed to add `children`");
        }
    }
}