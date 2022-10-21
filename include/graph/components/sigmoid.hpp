#pragma once
#include "base.hpp"
#include "../node/sigmoid.hpp"
#include <memory>


namespace aedlf {
    namespace components {
        template <typename MType>
        class Sigmoid : public BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::shared_ptr<std::vector<node_ptr>>;
                using matrix_dim = std::vector<unsigned long>;
                ~Sigmoid();
                Sigmoid(std::string layer_name);
                void construct(node_ptr_c input_node_c) override;
                node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) override;
                node_ptr_c operator()(node_ptr_c input_node_c) override;
                void backward(node_ptr end) override;
                void forward() override;
                Matrix<MType> get_data() override;
                void clear_jacobi() override;
            protected:
                node_ptr activate_node;
        };

        template <typename MType>
        Sigmoid<MType>::~Sigmoid() {

        }

        template <typename MType>
        Sigmoid<MType>::Sigmoid(std::string layer_name) {
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        void Sigmoid<MType>::construct(node_ptr_c input_node_c) {
            node_ptr input = input_node_c->at(0);
            matrix_dim c_dim {1,1,1,1};
            activate_node = std::make_shared<graph::SigmoidNode<MType>>(
                BaseComponent<MType>::layer_name + "_CORE",
                c_dim
            );
            activate_node->add_parent(input);
            BaseComponent<MType>::in_c = input_node_c;
            BaseComponent<MType>::out_c->push_back(activate_node);
            BaseComponent<MType>::complete_construct = true;
        }

        template <typename MType>
        typename Sigmoid<MType>::node_ptr_c Sigmoid<MType>::operator()(node_ptr_c input_node_c) {
            if(!BaseComponent<MType>::complete_construct) {
                construct(input_node_c);
            }
            return BaseComponent<MType>::out_c;
        }

        template <typename MType>
        typename Sigmoid<MType>::node_ptr_c Sigmoid<MType>::operator()(std::initializer_list<node_ptr> input_node_c) {
            node_ptr_c arg_wrapper {std::make_shared<std::vector<node_ptr>>(std::vector<node_ptr> {input_node_c})};
            return operator()(arg_wrapper);
        }

        template <typename MType>
        void Sigmoid<MType>::forward() {
            activate_node->forward();
        }

        template <typename MType>
        void Sigmoid<MType>::backward(node_ptr end) {
            BaseComponent<MType>::in_c->at(0)->backward(end);
            activate_node->backward(end);
        }

        template <typename MType>
        void Sigmoid<MType>::clear_jacobi() {
            activate_node->clear_jacobi();
        }

        template <typename MType>
        Matrix<MType> Sigmoid<MType>::get_data() {
            return activate_node->get_data();
        }
    }
}