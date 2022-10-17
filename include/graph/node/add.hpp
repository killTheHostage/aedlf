#pragma once
#include "common/base.hpp"


namespace aedlf {
    namespace graph {
        template <typename MType>
        class AddNode : public BaseNode<MType> {
            public:
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using BaseNode<MType>::BaseNode;
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
        };

        template <typename MType>
        void AddNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len >= 2);
            BaseNode<MType>::data.copy_from(static_cast<const Matrix<MType>>(BaseNode<MType>::get_parent(0)->get_data()));
            for(size_t i {1}; i < parents_len; ++i) {
                BaseNode<MType>::data += BaseNode<MType>::get_parent(i)->get_data();
            }
        }

        template <typename MType>
        void AddNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            typename BaseNode<MType>::matrix_dim jacobi_dim {BaseNode<MType>::data.get_dim()};
            int new_jacobi_dim = jacobi_dim[2] * jacobi_dim[3];
            jacobi_dim[2] = new_jacobi_dim;
            jacobi_dim[3] = new_jacobi_dim;
            matrix_tools::MakeMatrix<MType> mm {jacobi_dim};
            mm.identity(m);
        }
    }
}