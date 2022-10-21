#pragma once
#include "base.hpp"
#include "../node/logloss.hpp"
#include <memory>


namespace aedlf {
    namespace components {
        template <typename MType>
        class LogLoss : public BaseComponent<MType> {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                using node_ptr_c = std::shared_ptr<std::vector<node_ptr>>;
                using matrix_dim = std::vector<unsigned long>;
                ~LogLoss();
                LogLoss(std::string layer_name);
                void construct(node_ptr_c input_node_c) override;
                node_ptr_c operator()(std::initializer_list<node_ptr> input_node_c) override;
                node_ptr_c operator()(node_ptr_c input_node_c) override;
                void backward(node_ptr end) override;
                void forward() override;
                Matrix<MType> get_data() override;
                void clear_jacobi() override;
            protected:
                node_ptr loss_node;
        };

        template <typename MType>
        LogLoss<MType>::~LogLoss() {

        }

        template <typename MType>
        LogLoss<MType>::LogLoss(std::string layer_name) {
            BaseComponent<MType>::layer_name = layer_name;
        }

        template <typename MType>
        void LogLoss<MType>::construct(node_ptr_c input_node_c) {
            // label in index 1, pred in index 0
            node_ptr input = input_node_c->at(0);
            node_ptr label = input_node_c->at(1);
            matrix_dim c_dim {1,1,1,1};
            loss_node = std::make_shared<graph::LogLossNode<MType>>(
                BaseComponent<MType>::layer_name + "_CORE",
                c_dim
            );
            loss_node->add_parent(input);
            loss_node->add_parent(label);
            BaseComponent<MType>::in_c = input_node_c;
            BaseComponent<MType>::out_c->push_back(loss_node);
            BaseComponent<MType>::complete_construct = true;
        }

        template <typename MType>
        typename LogLoss<MType>::node_ptr_c LogLoss<MType>::operator()(node_ptr_c input_node_c) {
            if(!BaseComponent<MType>::complete_construct) {
                construct(input_node_c);
            }
            return BaseComponent<MType>::out_c;
        }

        template <typename MType>
        typename LogLoss<MType>::node_ptr_c LogLoss<MType>::operator()(std::initializer_list<node_ptr> input_node_c) {
            node_ptr_c arg_wrapper {std::make_shared<std::vector<node_ptr>>(std::vector<node_ptr> {input_node_c})};
            return operator()(arg_wrapper);
        }

        template <typename MType>
        void LogLoss<MType>::forward() {
            loss_node->forward();
        }

        template <typename MType>
        void LogLoss<MType>::backward(node_ptr end) {
            BaseComponent<MType>::in_c->at(0)->backward(end);
        }

        template <typename MType>
        void LogLoss<MType>::clear_jacobi() {
            loss_node->clear_jacobi();
        }

        template <typename MType>
        Matrix<MType> LogLoss<MType>::get_data() {
            return loss_node->get_data();
        }
    }
}