#pragma once
#include "./node/base.hpp"
#include "../math/matrix.hpp"
#include <cstddef>
#include <stdexcept>
#include <set>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class Graph {
            public:
                using node_ptr = std::shared_ptr<BaseNode<MType>>;
                using node_ptr_c = std::vector<std::shared_ptr<BaseNode<MType>>>;
                virtual ~Graph();
                Graph() {};
                virtual void forward(std::initializer_list<Matrix<MType>> input_m) = 0; //常规前传
                void clear_jacobi();
            protected:
                
        };
    }
}