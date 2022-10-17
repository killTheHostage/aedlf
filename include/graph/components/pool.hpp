#pragma once
#include "base.hpp"
#include "../node/pool.hpp"
#include <initializer_list>
#include <memory>
#include <stdexcept>


namespace aedlf {
    namespace components {
        template <typename MType>
        class MaxPool2d : BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::vector<node_ptr>;
                using matrix_dim = std::vector<int>;
                using kernel_shape = std::vector<unsigned>;
                ~MaxPool2d();
                MaxPool2d(std::string layer_name, unsigned kernel_size, unsigned padding, unsigned stride=0, MType padding_init=0);
                MaxPool2d(std::string layer_name, std::initializer_list<unsigned> kernel_size, std::initializer_list<unsigned> padding, unsigned stride=0, MType padding_init=0);
                MaxPool2d(std::string layer_name, kernel_shape kernel_size, kernel_shape padding, unsigned stride=0, MType padding_init=0);
                void construct(node_ptr_c input_node_c) override;
                node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) override;
                node_ptr_c operator()(node_ptr_c input_node_c) override;
                void backward(node_ptr end) override;
                void forward() override;
                Matrix<MType> get_data() override;
                void clear_jacobi() override;
            protected:
                node_ptr maxpool_node;
                kernel_shape kernel_size;
                kernel_shape padding_size;
                unsigned stride;
                MType padding_init;
                int output_h;
                int output_w;
        };

        template <typename MType>
        MaxPool2d<MType>::MaxPool2d(std::string layer_name, unsigned kernel_size, unsigned padding, unsigned stride, MType padding_init) {
            this->kernel_size = kernel_shape {kernel_size, kernel_size};
            this->padding_size = kernel_shape {padding, padding};
            this->stride = stride;
            this->padding_init = padding_init;
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        MaxPool2d<MType>::MaxPool2d(std::string layer_name, std::initializer_list<unsigned> kernel_size, std::initializer_list<unsigned> padding, unsigned stride, MType padding_init) {
            assert(kernel_size.size() == 2);
            assert(padding.size() == 2);
            this->kernel_size = kernel_shape {kernel_size};
            this->padding_size = kernel_shape {padding};
            this->stride = stride;
            this->padding_init = padding_init;
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        MaxPool2d<MType>::MaxPool2d(std::string layer_name, kernel_shape kernel_size, kernel_shape padding, unsigned stride, MType padding_init) {
            assert(kernel_size.size() == 2);
            assert(kernel_size.size() == 2);
            this->kernel_size = kernel_size;
            this->padding_size = padding;
            this->stride = stride;
            this->padding_init = padding_init;
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        void MaxPool2d<MType>::construct(node_ptr_c input_node_c) {
            assert(input_node_c.size() == 1);
            node_ptr input = input_node_c[0];
            matrix_dim c_dim {input->get_data_dim()};
            matrix_tools::MakeMatrix<MType> mm;
            Matrix<MType> data_padded {{1,1,1,1}, 0};
            if(stride == 0 && kernel_size[0] == kernel_size[1]){
                stride = kernel_size[0];
            }
            else if(stride == 0 && kernel_size[0] != kernel_size[1]) {
                throw std::runtime_error("When stride use default value, the padding must like NxN");
            }
            output_h = int(std::max(std::floor((c_dim[2] + 2 * padding_size[0] - kernel_size[0] - 2) / stride + 1), 0.0));
            output_w = int(std::max(std::floor((c_dim[3] + 2 * padding_size[1] - kernel_size[1] - 2) / stride + 1), 0.0));
            assert(output_h > 0 && output_w > 0);
            mm.add_padding(input->get_data(), data_padded, padding_size, padding_init);
            c_dim[2] = output_h;
            c_dim[3] = output_w;
            maxpool_node = std::make_shared<graph::MaxPool2dNode<MType>>(
                graph::MaxPool2dNode<MType> {BaseComponent<MType>::layer_name + "_CORE", c_dim}
            );
            // connect graph
            maxpool_node->add_parent(input);
            BaseComponent<MType>::in_c->input_node_c;
            BaseComponent<MType>::out_c->push_back(maxpool_node);
            BaseComponent<MType>::complete_construct = true;
        }

        template <typename MType>
        MaxPool2d<MType>::~MaxPool2d() {

        }

        template <typename MType>
        Matrix<MType> MaxPool2d<MType>::get_data() {
            return maxpool_node->get_data();
        }

        template <typename MType>
        void MaxPool2d<MType>::clear_jacobi() {
            maxpool_node->clear_jacobi();
        }

        template <typename MType>
        typename MaxPool2d<MType>::node_ptr_c MaxPool2d<MType>::operator()(node_ptr_c input_node_c) {
            if(!BaseComponent<MType>::complete_construct) {
                construct(input_node_c);
            }
            return node_ptr_c {maxpool_node};
        }

        template <typename MType>
        typename MaxPool2d<MType>::node_ptr_c MaxPool2d<MType>::operator()(std::initializer_list<node_ptr> input_node_c) {
            return operator()(node_ptr_c {input_node_c});
        }

        template <typename MType>
        void MaxPool2d<MType>::backward(node_ptr end) {
            BaseComponent<MType>::in_c[0]->backward(end);
        }

        template <typename MType>
        void MaxPool2d<MType>::forward() {
            maxpool_node->forward();
        }
    }
}