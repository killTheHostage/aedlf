#pragma once
#include "base.hpp"
#include "../node/data.hpp"
#include <initializer_list>
#include <memory>
#include <stdexcept>


namespace aedlf {
    namespace components {
        template <typename MType>
        class Data : public BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::shared_ptr<std::vector<node_ptr>>;
                using matrix_dim = std::vector<unsigned long>;
                ~Data();
                Data(std::string layer_name);
                void construct(Matrix<MType>& matrix);
                void construct(node_ptr_c input_node_c) override {};
                node_ptr_c operator()(Matrix<MType>& matrix);
                node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) override;
                node_ptr_c operator()(node_ptr_c input_node_c) override;
                void backward(node_ptr end) override;
                void forward() override {};
                Matrix<MType> get_data() override;
                void clear_jacobi() override {};
            protected:
                node_ptr data_node;
        };

        template <typename MType>
        Data<MType>::~Data<MType>() {

        }

        template <typename MType>
        Data<MType>::Data(std::string layer_name) {
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        void Data<MType>::construct(Matrix<MType>& matrix) {
            data_node = std::make_shared<graph::DataNode<MType>>(
                BaseComponent<MType>::layer_name + "_INDATA",
                matrix
            );

            BaseComponent<MType>::out_c->push_back(data_node);
            BaseComponent<MType>::complete_construct = true;
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
            return BaseComponent<MType>::out_c;
        }

        template <typename MType>
        typename Data<MType>::node_ptr_c Data<MType>::operator()(node_ptr_c input_node_c) {
            throw std::runtime_error("Component `Data` can only call operator()(Matrix<MType>& matrix)");
        }

        template <typename MType>
        typename Data<MType>::node_ptr_c Data<MType>::operator()(std::initializer_list<node_ptr> input_node_c) {
            throw std::runtime_error("Component `Data` can only call operator()(Matrix<MType>& matrix)");
        }

        template <typename MType>
        void Data<MType>::backward(node_ptr end) {
            data_node->backward(end);
        }
    }
}