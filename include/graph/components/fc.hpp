#pragma once
#include "base.hpp"
#include "../node/weight.hpp"
#include "../node/mul.hpp"
#include "../node/add.hpp"
#include "../node/data.hpp"
#include <vector>
#include <memory>
#include <initializer_list>
#include <string>
#include <map>


namespace aedlf {
    namespace components {
        template <typename MType>
        class FC : BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::vector<node_ptr>;
                using matrix_dim = std::vector<int>;
                ~FC();
                FC(std::string layer_name, unsigned output_dim, std::string weight_init="gaussian", std::string bias_init="zeros") : BaseComponent<MType>::layer_name(layer_name), output_dim(output_dim), weight_init(weight_init), bias_init(bias_init) {};
                void construct(node_ptr_c input_node_c) override;
                node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) override;
                node_ptr_c operator()(node_ptr_c input_node_c) override;
                void backward(node_ptr end) override;
                void forward() override;
                Matrix<MType> get_data() override;
                void clear_jacobi() override;
            protected:
                node_ptr weight_node;
                node_ptr mul_node;
                node_ptr bias_node;
                node_ptr add_node;
                unsigned output_dim;
                std::string weight_init;
                std::string bias_init;
        };

        template <typename MType>
        void FC<MType>::construct(node_ptr_c input_node_c) {
            node_ptr input = input_node_c[0];
            matrix_dim c_dim {input->get_data_dim()};
            assert(c_dim[2] == 1 && c_dim[3] == 1); // 必须是[x, x, 1, 1]这样的shape
            // init weight
            c_dim[2] = c_dim[3];
            c_dim[3] = output_dim;
            weight_node = std::make_shared<graph::WeightNode<MType>>(
                graph::WeightNode<MType> {BaseComponent<MType>::layer_name + "_WEIGHT", c_dim}
            );
            weight_node->init_data(weight_init);
            // init bias
            c_dim[2] = 1;
            c_dim[3] = output_dim;
            bias_node = std::make_shared<graph::WeightNode<MType>>(
                graph::WeightNode<MType> {BaseComponent<MType>::layer_name + "_BIAS", c_dim}
            );
            bias_node->init_data(bias_init);
            mul_node = std::make_shared<graph::MulNode<MType>>(
                graph::MulNode<MType> {BaseComponent<MType>::layer_name + "_MUL", c_dim}
            );
            add_node = std::make_shared<graph::AddNode<MType>>(
                graph::AddNode<MType> {BaseComponent<MType>::layer_name + "_ADD", c_dim}
            );
            mul_node->add_parent(weight_node);
            mul_node->add_parent(input);
            add_node->add_parent(mul_node);
            add_node->add_parent(bias_node);
            BaseComponent<MType>::in_c = input_node_c;
            BaseComponent<MType>::out_c->push_back(add_node);
            BaseComponent<MType>::complete_construct = true;
        }

        template <typename MType>
        FC<MType>::~FC() {
            add_node.reset();
            bias_node.reset();
            mul_node.reset();
            weight_node.reset();
        }

        template <typename MType>
        Matrix<MType> FC<MType>::get_data() {
            return add_node->get_data();
        }

        template <typename MType>
        void FC<MType>::clear_jacobi() {
            weight_node->clear_jacobi();
            mul_node->clear_jacobi();
            bias_node->clear_jacobi();
            add_node->clear_jacobi();
        }

        template <typename MType>
        typename FC<MType>::node_ptr_c FC<MType>::operator()(node_ptr_c input_node_c) {
            if(!BaseComponent<MType>::complete_construct) {
                construct(input_node_c);
            }
            return node_ptr_c {add_node};
        }

        template <typename MType>
        typename FC<MType>::node_ptr_c FC<MType>::operator()(std::initializer_list<node_ptr> input_node_c) {
            return operator()(node_ptr_c {input_node_c});
        }

        template <typename MType>
        void FC<MType>::backward(node_ptr end) {
            BaseComponent<MType>::in_c[0]->backward(end);
            weight_node->backward(end);
            bias_node->backward(end);
        }

        template <typename MType>
        void FC<MType>::forward() {
            add_node->forward();
        }
    }
}