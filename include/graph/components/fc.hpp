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
        class FC : public BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::shared_ptr<std::vector<node_ptr>>;
                using matrix_dim = std::vector<unsigned long>;
                ~FC();
                FC(std::string layer_name, unsigned long input_dim, unsigned long output_dim, std::string weight_init="gaussian", std::string bias_init="zeros");
                void construct(node_ptr_c input_node_c) override;
                node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) override;
                node_ptr_c operator()(node_ptr_c input_node_c) override;
                void backward(node_ptr end) override;
                void forward() override;
                void update(MType lr) override;
                Matrix<MType> get_data() override;
                void clear_jacobi() override;
            protected:
                node_ptr weight_node;
                node_ptr mul_node;
                node_ptr bias_node;
                node_ptr add_node;
                unsigned long output_dim_;
                unsigned long input_dim_;
                std::string weight_init_;
                std::string bias_init_;
                matrix_dim origin_dim;
        };

        template <typename MType>
        FC<MType>::FC(std::string layer_name, unsigned long input_dim, unsigned long output_dim, std::string weight_init, std::string bias_init) {
            input_dim_ = input_dim;
            output_dim_ = output_dim;
            weight_init_ = weight_init;
            bias_init_ = bias_init;
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        void FC<MType>::construct(node_ptr_c input_node_c) {
            node_ptr input = input_node_c->at(0);
            matrix_dim c_dim {input->get_data_dim()};
            // assert(c_dim[2] == 1 && c_dim[3] == 1); // 必须是[x, x, 1, 1]这样的shape
            assert(c_dim[1] * c_dim[2] * c_dim[3] == input_dim_);
            if(c_dim[1] != 1 || c_dim[3] != 1) {
                origin_dim = c_dim;
                input->view_data({c_dim[0], 1, input_dim_, 1});
            }
            // modify input_data 
            // init weight
            c_dim[2] = output_dim_;
            c_dim[3] = input_dim_;
            weight_node = std::make_shared<graph::WeightNode<MType>>(
                BaseComponent<MType>::layer_name + "_WEIGHT",
                c_dim
            );
            weight_node->init_data(weight_init_);
            // init bias
            c_dim[2] = output_dim_;
            c_dim[3] = 1;
            bias_node = std::make_shared<graph::WeightNode<MType>>(
                BaseComponent<MType>::layer_name + "_BIAS",
                c_dim
            );
            bias_node->init_data(bias_init_);
            mul_node = std::make_shared<graph::MulNode<MType>>(
                BaseComponent<MType>::layer_name + "_MUL",
                c_dim
            );
            add_node = std::make_shared<graph::AddNode<MType>>(
                BaseComponent<MType>::layer_name + "_ADD",
                c_dim
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
            return BaseComponent<MType>::out_c;
        }

        template <typename MType>
        typename FC<MType>::node_ptr_c FC<MType>::operator()(std::initializer_list<node_ptr> input_node_c) {
            node_ptr_c arg_wrapper {std::make_shared<std::vector<node_ptr>>(input_node_c)};
            return operator()(arg_wrapper);
        }

        template <typename MType>
        void FC<MType>::backward(node_ptr end) {
            weight_node->backward(end);
            bias_node->backward(end);
            weight_node->view_jacobi(origin_dim);
            BaseComponent<MType>::in_c->at(0)->backward(end);
        }

        template <typename MType>
        void FC<MType>::forward() {
            add_node->forward();
        }

        template <typename MType>
        void FC<MType>::update(MType lr) {
            weight_node->update(lr);
            bias_node->update(lr);
        }
    }
}