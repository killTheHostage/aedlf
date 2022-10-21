#pragma once
#include "common/base.hpp"


namespace aedlf {
    namespace graph {
        template <typename MType>
        class MulNode : public BaseNode<MType> {
            public:
                using ul_pos = const std::pair<unsigned long, unsigned long>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using BaseNode<MType>::BaseNode;
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
        };

        template <typename MType>
        void MulNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len >= 2);
            BaseNode<MType>::data.copy_from(BaseNode<MType>::get_parent(0)->get_data());
            for(size_t i {1}; i < parents_len; ++i) {
                BaseNode<MType>::data *= BaseNode<MType>::get_parent(i)->get_data();
            }
        }

        template <typename MType>
        void MulNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            // 注意，这里这个写法只能实现两个节点相乘的反向传播
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 2);
            matrix_tools::MakeMatrix<MType> mm {};
            // jacobi_dim is transpose dim
            typename BaseNode<MType>::matrix_dim jacobi_dim {BaseNode<MType>::data.get_dim()};
            if(parent_node == BaseNode<MType>::parents->at(0)) {
                typename BaseNode<MType>::matrix_dim parent0_dim {BaseNode<MType>::parents->at(0)->get_data().get_dim()};
                typename BaseNode<MType>::matrix_dim parent1_dim {BaseNode<MType>::parents->at(1)->get_data().get_dim()};
                jacobi_dim[2] = parent0_dim[2] * parent1_dim[3];
                jacobi_dim[3] = parent0_dim[2] * parent0_dim[3];
                mm.modify_dim(jacobi_dim);
                Matrix<MType> fw_m {BaseNode<MType>::parents->at(1)->get_data()};
                fw_m.T();
                mm.diagonal(m, fw_m);
            }
            else {
                typename BaseNode<MType>::matrix_dim parent2_dim {BaseNode<MType>::parents->at(1)->get_data().get_dim()};
                typename BaseNode<MType>::matrix_dim parent1_dim {BaseNode<MType>::parents->at(0)->get_data().get_dim()};
                jacobi_dim[2] = parent1_dim[2] * parent2_dim[3];
                jacobi_dim[3] = parent1_dim[3] * parent2_dim[3];
                mm.modify_dim(jacobi_dim);
                mm.special_jacobi(m, BaseNode<MType>::parents->at(0)->get_data(), parent2_dim[3]);
            }
        }
    }
}