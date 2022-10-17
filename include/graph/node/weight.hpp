#pragma once
#include "common/base.hpp"


namespace aedlf {
    namespace graph {
        template <typename MType>
        class WeightNode : public BaseNode<MType> {
            public:
                using BaseNode<MType>::BaseNode;
                void init_data(std::string init_method) override;
        };

        template <typename MType>
        void WeightNode<MType>::init_data(std::string init_method) {
            matrix_tools::MakeMatrix<MType> mm {BaseNode<MType>::data.get_dim()};
            std::map<std::string, void (matrix_tools::MakeMatrix<MType>::*)(Matrix<MType>&)> function_map {
                {std::string("kaiming"), &matrix_tools::MakeMatrix<MType>::kaiming},
                {std::string("gaussian"), &matrix_tools::MakeMatrix<MType>::gaussian},
                {std::string("xavier"), &matrix_tools::MakeMatrix<MType>::xavier},
                {std::string("ones"), &matrix_tools::MakeMatrix<MType>::ones},
                {std::string("zeros"), &matrix_tools::MakeMatrix<MType>::zeros}
            };
            auto func_map_iter = function_map.find(init_method);
            assert(func_map_iter != function_map.end());
            (mm.*function_map[init_method])(BaseNode<MType>::data);
        }
    }
}