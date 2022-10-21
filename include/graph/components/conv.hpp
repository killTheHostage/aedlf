#pragma once
#include "base.hpp"
#include "../node/weight.hpp"
#include "../node/mul.hpp"
#include "../node/add.hpp"
#include "../node/data.hpp"
#include "../node/conv.hpp"
#include "../node/padding.hpp"
#include "../node/img2col.hpp"
#include "../node/concat.hpp"
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <initializer_list>


namespace aedlf {
    namespace components {
        template <typename MType>
        class Conv2d : public BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::shared_ptr<std::vector<node_ptr>>;
                using matrix_dim = std::vector<unsigned long>;
                using kernel_shape = std::vector<unsigned long>;
                ~Conv2d();
                Conv2d(std::string layer_name, unsigned long in_channel, unsigned long output_channel, unsigned long kernel_size, unsigned long padding, unsigned long stride=1, std::string weight_init="gaussian", std::string bias_init="zeros", MType padding_init=0);
                Conv2d(std::string layer_name, unsigned long in_channel, unsigned long output_channel, std::initializer_list<unsigned long> kernel_size, std::initializer_list<unsigned long> padding, unsigned long stride=1, std::string weight_init="gaussian", std::string bias_init="zeros", MType padding_init=0);
                Conv2d(std::string layer_name, unsigned long in_channel, unsigned long output_channel, kernel_shape kernel_size, kernel_shape padding, unsigned long stride=1, std::string weight_init="gaussian", std::string bias_init="zeros", MType padding_init=0);
                void construct(node_ptr_c input_node_c) override;
                node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) override;
                node_ptr_c operator()(node_ptr_c input_node_c) override;
                void backward(node_ptr end) override;
                void forward() override;
                void update(MType lr) override;
                Matrix<MType> get_output() override;
                void clear_jacobi() override;
            protected:
                node_ptr weight_node;
                node_ptr bias_node;
                node_ptr conv_node;
                node_ptr padding_node;
                node_ptr img2col_node;
                node_ptr concat_node;
                unsigned long output_h;
                unsigned long output_w;
                unsigned long in_channel;
                unsigned long output_channel;
                kernel_shape kernel_size;
                kernel_shape padding_size;
                unsigned long stride;
                std::string weight_init;
                std::string bias_init;
                MType padding_init;
        };

        template <typename MType>
        Conv2d<MType>::~Conv2d() {

        }

        template <typename MType>
        Conv2d<MType>::Conv2d(std::string layer_name, unsigned long in_channel, unsigned long output_channel, unsigned long kernel_size, unsigned long padding, unsigned long stride, std::string weight_init, std::string bias_init, MType padding_init) {
            this->in_channel = in_channel;
            this->output_channel = output_channel;
            this->kernel_size = kernel_shape {kernel_size, kernel_size};
            this-padding_size = kernel_shape {padding, padding};
            this->stride = stride;
            this->weight_init = weight_init;
            this->bias_init = bias_init;
            this->padding_init = padding_init;
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        Conv2d<MType>::Conv2d(std::string layer_name, unsigned long in_channel, unsigned long output_channel, std::initializer_list<unsigned long> kernel_size, std::initializer_list<unsigned long> padding, unsigned long stride, std::string weight_init, std::string bias_init, MType padding_init) {
            this->in_channel = in_channel;
            this->output_channel = output_channel;
            this->kernel_size = kernel_shape {kernel_size};
            this-padding_size = kernel_shape {padding};
            this->stride = stride;
            this->weight_init = weight_init;
            this->bias_init = bias_init;
            this->padding_init = padding_init;
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        Conv2d<MType>::Conv2d(std::string layer_name, unsigned long in_channel, unsigned long output_channel, kernel_shape kernel_size, kernel_shape padding, unsigned long stride, std::string weight_init, std::string bias_init, MType padding_init) {
            this->in_channel = in_channel;
            this->output_channel = output_channel;
            this->kernel_size = kernel_size;
            this-padding_size = padding;
            this->stride = stride;
            this->weight_init = weight_init;
            this->bias_init = bias_init;
            this->padding_init = padding_init;
            BaseComponent<MType>::layer_name = layer_name;
        }
        

        template <typename MType>
        void Conv2d<MType>::construct(node_ptr_c input_node_c) {
            node_ptr input {input_node_c->at(0)};
            matrix_dim c_dim {input->get_data_dim()};
            matrix_tools::MakeMatrix<MType> mm;
            output_h = int(std::max(std::floor((c_dim[2] + 2 * padding_size[0] - kernel_size[0] - 2) / stride + 1), 0.0));
            output_w = int(std::max(std::floor((c_dim[3] + 2 * padding_size[1] - kernel_size[1] - 2) / stride + 1), 0.0));
            assert(output_h > 0 && output_w > 0);
            // init padding
            c_dim[2] = c_dim[2] + 2 * padding_size[0];
            c_dim[3] = c_dim[3] + 2 * padding_size[1];
            padding_node = std::make_shared<graph::PaddingNode<MType>>(
                BaseComponent<MType>::layer_name + "_PADDING",
                c_dim
            );
            // init img2col
            c_dim[2] = kernel_size[0] * kernel_size[1];
            c_dim[3] = output_h * output_w;
            img2col_node = std::make_shared<graph::Img2colNode<MType>>(
                BaseComponent<MType>::layer_name + "_IMG2COL",
                c_dim
            );
            // init weight
            c_dim[1] = in_channel;
            c_dim[2] = output_channel;
            c_dim[3] = kernel_size[0] * kernel_size[1];
            weight_node = std::make_shared<graph::WeightNode<MType>>(
                BaseComponent<MType>::layer_name + "_WEIGHT",
                c_dim
            );
            weight_node->init_data(weight_init);
            // init bias
            c_dim[1] = 1;
            c_dim[2] = output_channel;
            c_dim[3] = output_h * output_w;
            bias_node = std::make_shared<graph::MulNode<MType>>(
                BaseComponent<MType>::layer_name + "_BIAS",
                c_dim
            );
            // init conv
            conv_node = std::make_shared<graph::Conv2dNode<MType>>(
                BaseComponent<MType>::layer_name + "_CORE",
                c_dim
            );
            // connect graph
            padding_node->add_parent(input);
            img2col_node->add_parent(padding_node);
            conv_node->add_parent(weight_node);
            conv_node->add_parent(img2col_node);
            conv_node->add_parent(bias_node);
            BaseComponent<MType>::in_c = input_node_c;
            BaseComponent<MType>::out_c->push_back(conv_node);
            BaseComponent<MType>::complete_construct = true;
        }

        template <typename MType>
        typename Conv2d<MType>::node_ptr_c Conv2d<MType>::operator()(node_ptr_c input_node_c) {
            if(!BaseComponent<MType>::complete_construct) {
                construct(input_node_c);
            }
            return BaseComponent<MType>::out_c;
        }

        template <typename MType>
        typename Conv2d<MType>::node_ptr_c Conv2d<MType>::operator()(std::initializer_list<node_ptr> input_node_c) {
            node_ptr_c arg_wrapper {std::make_shared<std::vector<node_ptr>>(input_node_c)};
            return operator()(arg_wrapper);
        }

        template <typename MType>
        Matrix<MType> Conv2d<MType>::get_output() {
            return conv_node.get_data();
        }

        template <typename MType>
        void Conv2d<MType>::clear_jacobi() {
            weight_node->clear_jacobi();
            bias_node->clear_jacobi();
            padding_node->clear_jacobi();
            img2col_node->clear_jacobi();
            conv_node->clear_jacobi();
        }

        template <typename MType>
        void Conv2d<MType>::backward(node_ptr end) {
            BaseComponent<MType>::in_c->at(0)->backward(end);
            weight_node->backward(end);
            bias_node->backward(end);
        }

        template <typename MType>
        void Conv2d<MType>::forward() {
            conv_node->forward();
        }

        template <typename MType>
        void Conv2d<MType>::update(MType lr) {
            weight_node->update(lr);
            bias_node->update(lr);
        }
    }
}