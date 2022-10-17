#pragma once
#include "../graph/graph.hpp"
#include "../graph/node/base.hpp"
#include <memory>


namespace aedlf {
    namespace optim {
        template <typename MType>
        class Optimizer {
            public:
                using node_ptr = std::shared_ptr<graph::BaseNode<MType>>;
                Optimizer() {};
                Optimizer(graph::Graph<MType>& g);
                ~Optimizer();
                virtual void step();
                virtual void update_grad() = 0;
                void set_lr(double lr);
                void set_graph(graph::Graph<MType>& g);
            protected:
                double lr {0.0001};
                graph::Graph<MType> compute_graph {};
        };

        template <typename MType>
        Optimizer<MType>::Optimizer(graph::Graph<MType>& g) {
            compute_graph = g;
        }

        template <typename MType>
        void Optimizer<MType>::set_lr(double lr) {
            this->lr = lr;
        }

        template <typename MType>
        void Optimizer<MType>::set_graph(graph::Graph<MType> &g) {
            compute_graph = g;
        }

        template <typename MType>
        void Optimizer<MType>::step() {
            
        }
    }
}