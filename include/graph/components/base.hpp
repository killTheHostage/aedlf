#pragma once
#include "../node/common/base.hpp"
#include <vector>
#include <memory>
#include <initializer_list>


namespace aedlf {
    namespace components {
        template <typename MType>
        class BaseComponent {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::vector<node_ptr>;
                BaseComponent();
                virtual ~BaseComponent();
                virtual void forward() = 0;
                virtual void backward(node_ptr end) = 0;
                virtual void construct(node_ptr_c input_node_c) = 0;
                virtual node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) = 0;
                virtual node_ptr_c operator()(node_ptr_c input_node_c) = 0;
                virtual node_ptr_c get_output_nodes();
                virtual Matrix<MType> get_data() = 0;
                virtual void clear_jacobi() = 0;
            protected:
                node_ptr_c in_c;
                node_ptr_c out_c;
                std::string layer_name;
                bool complete_construct {false};
        };

        template <typename MType>
        typename BaseComponent<MType>::node_ptr_c BaseComponent<MType>::get_output_nodes() {
            return out_c;
        }
    }
}