#include "thrid-part/npy.hpp"
#include "include/math/matrix.hpp"
#include "include/math/tools.hpp"
#include "include/graph/components/data.hpp"
#include "include/graph/components/fc.hpp"
#include "include/graph/components/logloss.hpp"
#include "include/graph/components/sigmoid.hpp"
#include "include/utils/node_construct.hpp"
#include "include/utils/output.hpp"
#include <vector>
#include <memory>
#include <iostream>



int main() {
    using namespace aedlf;
    using node_ptr = std::shared_ptr<graph::BaseNode<double>>;
    using node_ptr_c = std::shared_ptr<std::vector<node_ptr>>;
    std::vector<double> train_data;
    std::vector<unsigned long> train_data_shape; // 200 4
    bool train_data_fortran;
    std::vector<double> train_label;
    std::vector<unsigned long> train_label_shape;
    bool train_label_fortran;
    npy::LoadArrayFromNumpy("./train_data.npy", train_data_shape, train_data_fortran, train_data);
    npy::LoadArrayFromNumpy("./train_label.npy", train_label_shape, train_label_fortran, train_label);
    Matrix<double> t_data {{50, 1, 1, 4}, std::make_shared<std::vector<double>>(train_data)};
    Matrix<double> t_label {{50, 1, 1, 1}, std::make_shared<std::vector<double>>(train_label)};
    // compute graph start
    node_ptr label_node {utils::construct_data_node("label_node", t_label)};
    components::Data<double> input_data {"data_layer"};
    components::FC<double> fc_layer {"mlp_layer", 4, 1, "ones"};
    components::LogLoss<double> loss_layer {"loss_layer"};
    components::Sigmoid<double> sigmoid_layer {"sigmoid_layer"};
    node_ptr_c i_data {input_data(t_data)};
    node_ptr_c fc_out {fc_layer(i_data)};
    node_ptr_c sigmoid_out {sigmoid_layer(fc_out)};
    sigmoid_out->push_back(label_node);
    node_ptr_c loss {loss_layer(sigmoid_out)};
    //compute graph end
    float lr {1e-4};
    for(int i {0}; i < 200; ++i) {
        loss_layer.forward();
        fc_layer.backward(loss->at(0));
        fc_layer.update(lr);
        input_data.clear_jacobi();
        fc_layer.clear_jacobi();
        loss_layer.clear_jacobi();
        sigmoid_layer.clear_jacobi();
        std::cout << "index: " << i << " loss: ";
        Matrix<double> loss_value {loss->at(0)->get_data()};
        utils::print_matrix<double>(loss_value);
    }
    return 0;
}