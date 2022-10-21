#pragma once
#include "../graph/node/data.hpp"
#include "../graph/node/common/base.hpp"
#include "../math/matrix.hpp"
#include <memory>


namespace aedlf {
    namespace utils {
        template <typename MType>
        std::shared_ptr<graph::BaseNode<MType>> construct_data_node(std::string node_name, Matrix<MType>& data_matrix) {
            return std::make_shared<graph::DataNode<MType>>(
                graph::DataNode<MType> {node_name + "_DATA_CONSTRUCT", data_matrix}
            );
        }
    }
}