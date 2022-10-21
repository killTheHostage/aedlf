#pragma once
#include "../../../math/matrix.hpp"
#include "../../../math/tools.hpp"
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <memory>
#include <string>
#include <random>
#include <map>
#include <thread>


namespace aedlf {
    namespace graph {
        template <typename MType>
        class BaseNode : public std::enable_shared_from_this<BaseNode<MType>>{
            public:
                using graph_nodes = std::shared_ptr<std::vector<std::shared_ptr<BaseNode>>>;
                using node_ptr = std::shared_ptr<BaseNode>;
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using matrix_p = std::shared_ptr<Matrix<MType>>;
                using matrix_dim = std::vector<unsigned long>;
                BaseNode(std::string node_name, matrix_dim m_dim);
                BaseNode(std::string node_name, const Matrix<MType>& m);
                BaseNode(std::string node_name, matrix_data_p data, matrix_dim m_dim);
                BaseNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent);
                BaseNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents);
                virtual ~BaseNode();
                virtual graph_nodes get_childrens();
                virtual graph_nodes get_parents();
                virtual node_ptr get_children(size_t children_id);
                virtual node_ptr get_parent(size_t parent_id);
                virtual size_t get_parents_len();
                virtual size_t get_childrens_len();
                virtual void forward();
                virtual void backward(node_ptr output_node);// 计算本节点的jacobi矩阵
                virtual void compute_forward() {};
                virtual void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {}; //计算子节点对当前节点的jacobi矩阵，一般在子节点（计算结果）上调用这个方法可以得到本节点对子节点的jacobi矩阵
                virtual void no_grad();
                virtual void ask_grad();
                virtual void add_parent(node_ptr parent);
                virtual void add_children(node_ptr children);
                virtual void set_data(const Matrix<MType>& m);
                virtual void clear_jacobi();
                virtual void update(MType lr) {};
                virtual Matrix<MType> get_data();
                virtual Matrix<MType> get_jacobi();
                virtual matrix_p get_m_data();
                virtual matrix_p get_m_jacobi();
                virtual matrix_dim get_data_dim();
                virtual matrix_dim get_jacobi_dim();
                virtual void view_data(matrix_dim shape);
                virtual void view_data(unsigned long n, unsigned long c, unsigned long h, unsigned long w);
                virtual void view_data(std::initializer_list<unsigned long> shape);
                virtual void view_jacobi(matrix_dim shape);
                virtual void view_jacobi(unsigned long n, unsigned long c, unsigned long h, unsigned long w);
                virtual void view_jacobi(std::initializer_list<unsigned long> shape);
                virtual void init_data(std::string init_method) {};
                bool is_jacobi_exists();
            protected:
                graph_nodes parents {std::make_shared<std::vector<std::shared_ptr<BaseNode>>>()};
                graph_nodes childrens {std::make_shared<std::vector<std::shared_ptr<BaseNode>>>()};
                std::string name;
                Matrix<MType> data {}; // 当前节点的数据
                Matrix<MType> jacobi {}; // 结果节点对本节点的jacobi矩阵
                bool wait_backward {false};
                bool require_grad {true};
        };

        template <typename MType>
        BaseNode<MType>::~BaseNode() {
            for(size_t child_i {0}; child_i < childrens->size(); ++child_i) {
                childrens->at(child_i).reset();
            }
            childrens.reset();
        }

        template <typename MType>
        BaseNode<MType>::BaseNode(std::string node_name, matrix_dim m_dim) {
            name = node_name;
            data = Matrix<MType>(m_dim, MType(0));
            matrix_dim empty_jacobi {1,1,1,1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
        }

        template <typename MType>
        BaseNode<MType>::BaseNode(std::string node_name, const Matrix<MType>& m) {
            name = node_name;
            data = m;
            matrix_dim empty_jacobi {1,1,1,1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
        }

        template <typename MType>
        BaseNode<MType>::BaseNode(std::string node_name, matrix_data_p data, matrix_dim m_dim) {
            name = node_name;
            data = Matrix<MType>(data, m_dim);
            matrix_dim empty_jacobi {1,1,1,1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
        }

        template <typename MType>
        BaseNode<MType>::BaseNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, node_ptr parent) {
            name = node_name;
            data = Matrix<MType>(data, m_dim);
            matrix_dim empty_jacobi {1,1,1,1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
            add_parent(parent);
            parent->add_children(std::make_shared<node_ptr>(this));
        }

        template <typename MType>
        BaseNode<MType>::BaseNode(std::string node_name, matrix_data_p data, matrix_dim m_dim, graph_nodes parents) {
            this->parents.reset();
            this->parents = parents;
            name = node_name;
            data = Matrix<MType>(data, m_dim);
            matrix_dim empty_jacobi {1,1,1,1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
            for(size_t parent_i {0}; parent_i < parents->size(); ++parent_i) {
                parents->at(parent_i)->add_children(std::make_shared<node_ptr>(this));
            }
        }

        template <typename MType>
        typename BaseNode<MType>::graph_nodes BaseNode<MType>::get_childrens() {
            return childrens;
        }

        template <typename MType>
        typename BaseNode<MType>::node_ptr BaseNode<MType>::get_children(size_t children_id) {
            if(childrens == nullptr) {
                return nullptr;
            }
            return childrens->at(children_id);
        }

        template <typename MType>
        typename BaseNode<MType>::graph_nodes BaseNode<MType>::get_parents() {
            return parents;
        }

        template <typename MType>
        typename BaseNode<MType>::node_ptr BaseNode<MType>::get_parent(size_t parent_id) {
            if(parents == nullptr) {
                return nullptr;
            }
            return parents->at(parent_id);
        }

        template <typename MType>
        size_t BaseNode<MType>::get_childrens_len() {
            if(childrens == nullptr) {
                return 0;
            }
            return childrens->size();
        }

        template <typename MType>
        size_t BaseNode<MType>::get_parents_len() {
            if(parents == nullptr) {
                return 0;
            }
            return parents->size();
        }

        template <typename MType>
        void BaseNode<MType>::add_parent(node_ptr parent) {
            if(parents == nullptr) {
                parents = std::make_shared<std::vector<std::shared_ptr<BaseNode>>>();
            }
            for(size_t i {0}; i < parents->size(); ++i) {
                if(parent == parents->at(i)) {
                    return;
                }
            }
            parents->push_back(parent);
            parent->add_children(std::enable_shared_from_this<BaseNode<MType>>::shared_from_this());
        }

        template <typename MType>
        void BaseNode<MType>::add_children(node_ptr children) {
            if(children == nullptr) {
                childrens = std::make_shared<std::vector<std::shared_ptr<BaseNode>>>();
            }
            for(size_t i {0}; i < childrens->size(); ++i) {
                if(children == childrens->at(i)) {
                    return;
                }
            }
            childrens->push_back(children);
            children->add_parent(std::enable_shared_from_this<BaseNode<MType>>::shared_from_this());
        }

        template <typename MType>
        void BaseNode<MType>::forward() {
            for(size_t parent_i {0}; parent_i < parents->size(); ++parent_i) {
                if(!parents->at(parent_i)->wait_backward) {
                    parents->at(parent_i)->forward();
                }
            }
            compute_forward();
            wait_backward = true;
        }

        template <typename MType>
        void BaseNode<MType>::backward(node_ptr output_node) {
            if(!require_grad) {
                for(size_t children_i {0}; children_i < childrens->size(); ++children_i) {
                    jacobi += childrens->at(children_i)->jacobi;
                }
                return;
            }
            if(std::enable_shared_from_this<BaseNode<MType>>::shared_from_this() == output_node) {
                matrix_dim jacobi_dim {BaseNode<MType>::data.get_dim()};
                unsigned long jacobi_shape {jacobi_dim[2] * jacobi_dim[3]};
                jacobi_dim[2] = jacobi_shape;
                jacobi_dim[3] = jacobi_shape;
                matrix_tools::MakeMatrix<MType> mm {jacobi_dim};
                mm.identity(BaseNode<MType>::jacobi);
            }
            matrix_dim temp_dim {1,1,1,1};
            matrix_dim children_dim {output_node->data.get_dim()};
            matrix_dim this_dim {data.get_dim()};
            Matrix<MType> temp {temp_dim, MType(0)};
            jacobi.resize(this_dim[0], this_dim[1], children_dim[2] * children_dim[3], this_dim[2] * this_dim[3], 0);
            for(size_t children_i {0}; children_i < childrens->size(); ++children_i) {
                if(childrens->at(children_i)->wait_backward) {
                    childrens->at(children_i)->backward(output_node);
                }
                childrens->at(children_i)->compute_jacobi(temp, std::enable_shared_from_this<BaseNode<MType>>::shared_from_this());
                matrix_dim child_jacobi_dim {childrens->at(children_i)->jacobi.get_dim()};
                matrix_dim temp_jacobi_dim {temp.get_dim()};
                if(child_jacobi_dim[2] == temp_jacobi_dim[3]) {
                    jacobi += childrens->at(children_i)->jacobi * temp;
                }
                else {
                    jacobi += temp.mul_v(childrens->at(children_i)->jacobi);
                }
                
            }
            jacobi.view(data.get_dim());
            wait_backward = false;
        }

        template <typename MType>
        void BaseNode<MType>::set_data(const Matrix<MType>& m) {
            data = m;
        }

        template <typename MType>
        void BaseNode<MType>::clear_jacobi() {
            jacobi.clear_data();
        }

        template <typename MType>
        void BaseNode<MType>::no_grad() {
            require_grad = false;
        }

        template <typename MType>
        void BaseNode<MType>::ask_grad() {
            require_grad = true;
        }

        template <typename MType>
        Matrix<MType> BaseNode<MType>::get_data() {
            return data;
        }

        template <typename MType>
        Matrix<MType> BaseNode<MType>::get_jacobi() {
            return jacobi;
        }

        template <typename MType>
        typename BaseNode<MType>::matrix_p BaseNode<MType>::get_m_data() {
            return std::make_shared<Matrix<MType>>(data);
        }

        template <typename MType>
        typename BaseNode<MType>::matrix_p BaseNode<MType>::get_m_jacobi() {
            return std::make_shared<Matrix<MType>>(jacobi);
        }

        template <typename MType>
        typename BaseNode<MType>::matrix_dim BaseNode<MType>::get_data_dim() {
            return data.get_dim();
        }

        template <typename MType>
        typename BaseNode<MType>::matrix_dim BaseNode<MType>::get_jacobi_dim() {
            return jacobi.get_dim();
        }

        template <typename MType>
        bool BaseNode<MType>::is_jacobi_exists() {
            return !jacobi.is_uninitialized();
        }

        template <typename MType>
        void BaseNode<MType>::view_data(matrix_dim shape) {
            data.view(shape);
        }

        template <typename MType>
        void BaseNode<MType>::view_data(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {
            data.view(n, c, h, w);
        }

        template <typename MType>
        void BaseNode<MType>::view_data(std::initializer_list<unsigned long> shape) {
            data.view(shape);
        }

        template <typename MType>
        void BaseNode<MType>::view_jacobi(matrix_dim shape) {
            jacobi.view(shape);
        }

        template <typename MType>
        void BaseNode<MType>::view_jacobi(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {
            jacobi.view(n, c, h, w);
        }

        template <typename MType>
        void BaseNode<MType>::view_jacobi(std::initializer_list<unsigned long> shape) {
            jacobi.view(shape);
        }
    }
}