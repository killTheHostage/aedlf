#pragma once
#include "common/loss.hpp"
#include <cstddef>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class LogLossNode : LossNode<MType> {
            public:
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode<MType>>>>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using matrix_dim = std::vector<int>;
                using BaseNode<MType>::BaseNode;
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
        };

        template <typename MType>
        void LogLossNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 2);
        }

        template <typename MType>
        void LogLossNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {

        }
    }
}