#include <iostream>
#include "../math/matrix.hpp"


namespace aedlf {
    namespace utils {
        template <typename MType>
        void print_matrix(Matrix<MType>& m) {
            const std::shared_ptr<std::vector<MType>> m_data_p = m.get_data();
            const std::vector<unsigned long> m_dim = m.get_dim();
            std::cout << "[";
            for(unsigned long n {0}; n < m_dim[0]; ++n) {
                std::cout << "[";
                for(unsigned long c {0}; c < m_dim[1]; ++c) {
                    std::cout << "[";
                    for(unsigned long h {0}; h < m_dim[2]; ++h) {
                        std::cout << "[";
                        for(unsigned long w {0}; w < m_dim[3]; ++w) {
                            std::cout << m_data_p->at(n * m_dim[1] * m_dim[2] * m_dim[3] + c * m_dim[2] * m_dim[3] + h * m_dim[3] + w);
                            if(w + 1 < m_dim[3]) {
                                std::cout << ",";
                            }
                        }
                        if(h + 1 < m_dim[2]) {
                            std::cout << "],";
                        }
                        else {
                            std::cout << "]";
                        }
                    }
                    if(c + 1 < m_dim[1]) {
                        std::cout << "],";
                    }
                    else {
                        std::cout << "]";
                    }
                }
                if(n + 1 < m_dim[0]) {
                    std::cout << "],";
                }
                else {
                    std::cout << "]";
                }
            }
            std::cout << "]" << std::endl;
        }
    }
}