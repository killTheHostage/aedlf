#pragma once
#include "base.hpp"
#include "../node/data.hpp"
#include <initializer_list>
#include <memory>
#include <stdexcept>


namespace aedlf {
    namespace components {
        template <typename MType>
        class Data : BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::vector<node_ptr>;
                using matrix_dim = std::vector<int>;
                using kernel_shape = std::vector<unsigned>;
                ~Data();
                Data(std::string layer_name);
                void construct(Matrix<MType>& matrix) override;
                node_ptr_c operator()(Matrix<MType>& matrix);
                void backward(node_ptr end) override;
                void forward() override {};
                Matrix<MType> get_data() override;
                void clear_jacobi() override {};
            protected:
                node_ptr data_node;
        };

        template <typename MType>
        Data<MType>::Data(std::string layer_name) {
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        void Data<MType>::construct(Matrix<MType>& matrix) {
            data_node = std::make_shared<graph::DataNode<MType>>(
                graph::DataNode<MType> {BaseComponent<MType>::layer_name + "_IMG2COL", matrix.get_dim()}
            );
            BaseComponent<MType>::complete_construct = true;
        }

        template <typename MType>
        Data<MType>::~Data() {

        }

        template <typename MType>
        Matrix<MType> Data<MType>::get_data() {
            return data_node->get_data();
        }

        template <typename MType>
        typename Data<MType>::node_ptr_c Data<MType>::operator()(Matrix<MType>& matrix) {
            if(!BaseComponent<MType>::complete_construct) {
                construct(matrix);
            }
            return node_ptr_c {data_node};
        }

        template <typename MType>
        void Data<MType>::backward(node_ptr end) {
            BaseComponent<MType>::in_c[0]->backward(end);
        }
    }
}