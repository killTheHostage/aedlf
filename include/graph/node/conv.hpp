#pragma once
#include "common/base.hpp"
#include <cstddef>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class Conv2dNode : public BaseNode<MType> {
            // backward需要定制
            public:
                using BaseNode<MType>::BaseNode;
                using ul_pos = const std::pair<unsigned long, unsigned long>;
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using matrix_dim = std::vector<unsigned long>;
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
        };

        template <typename MType>
        void Conv2dNode<MType>::compute_forward() {
            size_t parents_len {BaseNode<MType>::get_parents_len()};
            assert(parents_len == 3);
            /* 
            这里假设待卷积的图像已经经过了列变换，权重也已经更改过尺寸了
            权重尺寸是[n, c, output_channel, k_h * k_w]
            输入数据尺寸[n, c, k_h * k_w, output_h * output_w]
            bias数据尺寸[n, c, output_channel, output_h * output_w]
            强制规定wx+b parents [w, w, b]
            输出尺寸和bias相同
            */
            Matrix<MType> result {{1,1,1,1}, 0};
            matrix_dim result_dim {BaseNode<MType>::get_parent(2)->get_dim()};
            result = BaseNode<MType>::get_parent(0)->get_data() * BaseNode<MType>::get_parent(1)->get_data();
            result += BaseNode<MType>::get_parent(2)->get_data();
            result = result.sum_by_dim(1);
            BaseNode<MType>::data->copy_from(result);
        }

        template <typename MType>
        void Conv2dNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            /*
            这里的BP需要考虑，首先结果是concat出来的，反向传播需要先处理concat
            然后还有一个sum_by_dim，还需要处理sum_by_dim，最后才能处理乘法的BP
            然后不要忘了这里的compute_jacobi，计算的是子节点（也就是卷积的权重节点）对本节点的jacobi矩阵
            */
            matrix_tools::MakeMatrix<MType> mm {};
            matrix_dim jacobi_dim {BaseNode<MType>::data.get_dim()};
            if(parent_node == BaseNode<MType>::get_parent(0)) {
                matrix_dim parent1_dim {BaseNode<MType>::parents->at(0)->get_data().get_dim()};
                matrix_dim parent2_dim {BaseNode<MType>::parents->at(1)->get_data().get_dim()};
                jacobi_dim[3] = parent1_dim[2] * parent2_dim[3];
                jacobi_dim[2] = parent1_dim[2] * parent1_dim[3];
                mm.modify_dim(jacobi_dim);
                Matrix<MType> fw_m {BaseNode<MType>::parents->at(1)->get_data()};
                mm.diagonal(m, fw_m.T());
            }
            else if(parent_node == BaseNode<MType>::get_parent(1)) {
                matrix_dim parent2_dim {BaseNode<MType>::parents->at(1)->get_data().get_dim()};
                matrix_dim parent1_dim {BaseNode<MType>::parents->at(0)->get_data().get_dim()};
                jacobi_dim[2] = parent1_dim[2] * parent2_dim[3];
                jacobi_dim[3] = parent1_dim[3] * parent2_dim[3];
                mm.modify_dim(jacobi_dim);
                mm.special_jacobi(m, BaseNode<MType>::parents->at(0)->get_data(), parent2_dim[3]);
            }
            else if(parent_node == BaseNode<MType>::get_parent(2)) {
                int new_jacobi_dim = jacobi_dim[2] * jacobi_dim[3];
                jacobi_dim[2] = new_jacobi_dim;
                jacobi_dim[3] = new_jacobi_dim;
                mm.modify_dim(jacobi_dim);
                mm.identity(m);
            }
        }
    }
}