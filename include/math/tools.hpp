#pragma once
#include "matrix.hpp"
#include <cmath>
#include <cstddef>
#include <memory>
#include <pthread.h>
#include <vector>
#include <utility>
#include <random>
#include <thread>
#include <initializer_list>


namespace aedlf {
    namespace matrix_tools {
        template <typename MType>
        class MakeMatrix {
            public:
                using matrix_data_p = std::shared_ptr<std::vector<MType>>;
                using matrix_dim = std::vector<int>;
                using kernel_shape = std::vector<unsigned>;
                using ul_pos = const std::pair<int, int>;
                MakeMatrix() {};
                MakeMatrix(const matrix_dim dim) : m_dim(dim) {};
                MakeMatrix(std::initializer_list<int> dim_init) : m_dim(dim_init) {};
                void gaussian(Matrix<MType>& m);
                void xavier(Matrix<MType>& m);
                void kaiming(Matrix<MType>& m);
                void diagonal(Matrix<MType>& m, MType fill_with);
                void diagonal(Matrix<MType>& m, const Matrix<MType>& fill_with);
                void special_jacobi(Matrix<MType>& m, const Matrix<MType>& s_jacobi_fw, int jacobi_k);
                void zeros(Matrix<MType>& m);
                void ones(Matrix<MType>& m);
                void identity(Matrix<MType>& m);
                void add_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned padding, MType fill_with);
                void add_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, MType fill_with);
                void add_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned> padding, MType fill_with);
                void sub_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned padding);
                void sub_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding);
                void sub_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned> padding);
                void img2col(Matrix<MType>& m, Matrix<MType>& fw, unsigned kernel_size, unsigned stride);
                void img2col(Matrix<MType>& m, Matrix<MType>& fw, kernel_shape kernel_size, unsigned stride);
                void img2col(Matrix<MType>& m, Matrix<MType>& fw, std::initializer_list<unsigned> kernel_size, unsigned stride);
                void col2img(Matrix<MType>& m, Matrix<MType>& fw, unsigned kernel_size, unsigned stride, matrix_dim fw_dim);
                void col2img(Matrix<MType>& m, Matrix<MType>& fw, kernel_shape kernel_size, unsigned stride, matrix_dim fw_dim);
                void col2img(Matrix<MType>& m, Matrix<MType>& fw, std::initializer_list<unsigned> kernel_size, unsigned stride, matrix_dim fw_dim);
                void modify_dim(matrix_dim new_dim);
                void modify_dim(std::initializer_list<int> new_dim);
            private:
                void diagonal_core(Matrix<MType>& m, ul_pos m_channel_ul, const Matrix<MType>& fill_with, ul_pos fw_channel_ul, int h, int w);
                void special_jacobi_core(Matrix<MType>& m, ul_pos m_channel_ul, const Matrix<MType>& sjb, ul_pos fw_channel_ul, int jacobi_k);
                void add_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos m_channel_ul, ul_pos fw_channel_ul);
                void sub_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos m_channel_ul, ul_pos fw_channel_ul);
                void img2col_core(Matrix<MType>& m, Matrix<MType>& fw, ul_pos m_channel_ul, ul_pos fw_channel_ul, kernel_shape kernel_size, unsigned stride, unsigned output_h, unsigned output_w);
                void col2img_core(Matrix<MType>& m, Matrix<MType>& fw, ul_pos m_channel_ul, ul_pos fw_channel_ul, kernel_shape kernel_size, unsigned stride, unsigned output_h, unsigned output_w);
                matrix_dim m_dim;
        };

        template <typename MType>
        void MakeMatrix<MType>::modify_dim(matrix_dim new_dim) {
            m_dim = new_dim;
        }

        template <typename MType>
        void MakeMatrix<MType>::modify_dim(std::initializer_list<int> new_dim) {
            modify_dim(matrix_dim {new_dim});
        }

        template <typename MType>
        void MakeMatrix<MType>::gaussian(Matrix<MType>& m) {
            std::random_device gauss_rd {};
            std::mt19937 gauss_gen {gauss_rd()};
            std::normal_distribution<MType> gauss_d {0, 0.2};
            m.resize(m_dim, 0);
            matrix_data_p m_matrix_data = m.get_m_data();
            for(size_t i {0}; i < m_matrix_data->size(); ++i) {
                m_matrix_data->at(i) = MType(gauss_d(gauss_gen));
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::diagonal(Matrix<MType>& m, MType fill_with) {
            assert(m_dim[2] == m_dim[3]);
            m.resize(m_dim, 0);
            matrix_data_p m_matrix_data = m.get_m_data();
            int nxc = m_dim[0] * m_dim[1];
            int hxw = m_dim[2] * m_dim[3];
            for(int nc {0}; nc < nxc; ++nc) {
                for(int hw {0}; hw < m_dim[2]; ++hw) {
                    m_matrix_data->at(nc * hxw + hw * m_dim[2] + hw) = fill_with;
                }
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::diagonal(Matrix<MType>& m, const Matrix<MType>& fill_with) {
            matrix_dim fill_with_dim {fill_with.get_dim()};
            assert(m_dim[2] % fill_with_dim[2] == 0);
            assert(m_dim[3] % fill_with_dim[3] == 0);
            int n_h {m_dim[2] / fill_with_dim[2]};
            int n_w {m_dim[3] / fill_with_dim[3]};
            assert(n_h == n_w);
            m.resize(m_dim, 0);
            std::vector<std::thread> thread_c;
            for(int n {0}; n < m_dim[0]; ++n) {
                ul_pos m_batch_ul {m.get_batch(n)};
                ul_pos fw_batch_ul {fill_with.get_batch(n)};
                for(int c {0}; c < m_dim[1]; ++c) {
                    ul_pos m_channel_ul {m.get_channel(c, m_batch_ul)};
                    ul_pos fw_channel_ul {fill_with.get_channel(c, fw_batch_ul)};
                    for(int i {0}; i < n_h; ++i) {
                        int h {i * fill_with_dim[2]};
                        int w {i * fill_with_dim[3]};
                        thread_c.emplace_back(&MakeMatrix<MType>::diagonal_core, this, std::ref(m), m_channel_ul, std::ref(fill_with), fw_channel_ul, h, w);
                    }
                }
            }
            for (auto& t : thread_c) {
                t.join();
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::diagonal_core(Matrix<MType>& m, ul_pos m_channel_ul, const Matrix<MType>& fill_with, ul_pos fw_channel_ul, int h, int w) {
            // channel scale [ul_pos.first, ul_pos.second]
            matrix_dim fill_with_dim {fill_with.get_dim()};
            matrix_dim m_dim {m.get_dim()};
            matrix_data_p m_data_p = m.get_m_data();
            const matrix_data_p fw_data_p = fill_with.get_data();
            assert(h + fill_with_dim[2] <= m_dim[2]);
            assert(w + fill_with_dim[3] <= m_dim[3]);
            for(int inner_h {0}; inner_h < fill_with_dim[2]; ++inner_h) {
                for(int inner_w {0}; inner_w < fill_with_dim[3]; ++inner_w) {
                    m_data_p->at(m_channel_ul.first + (inner_h + h) * m_dim[3] + w + inner_w) = fw_data_p->at(fw_channel_ul.first + inner_h * fill_with_dim[3] + inner_w);
                }
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::special_jacobi(Matrix<MType> &m, const Matrix<MType> &s_jacobi_fw, int jacobi_k) {
            matrix_dim m_dim {m.get_dim()};
            matrix_dim sjfw_dim {s_jacobi_fw.get_dim()};
            assert(m_dim[0] == sjfw_dim[0]);
            assert(m_dim[1] == sjfw_dim[1]);
            m.resize(m_dim, 0);
            std::vector<std::thread> thread_c;
            for(int n {0}; n < m_dim[0]; ++n) {
                ul_pos m_batch_ul {m.get_batch(n)};
                ul_pos fw_batch_ul {s_jacobi_fw.get_batch(n)};
                for(int c {0}; c < m_dim[1]; ++c) {
                    ul_pos m_channel_ul {m.get_channel(c, m_batch_ul)};
                    ul_pos fw_channel_ul {s_jacobi_fw.get_channel(c, fw_batch_ul)};
                    thread_c.emplace_back(&MakeMatrix<MType>::special_jacobi_core, this, std::ref(m), m_channel_ul, std::ref(s_jacobi_fw), fw_channel_ul, jacobi_k);
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::special_jacobi_core(Matrix<MType>& m, ul_pos m_channel_ul, const Matrix<MType>& sjfw, ul_pos fw_channel_ul, int jacobi_k) {
            matrix_dim m_dim {m.get_dim()};
            matrix_dim sjfw_dim {sjfw.get_dim()};
            matrix_data_p m_data {m.get_m_data()};
            const matrix_data_p sjfw_data {sjfw.get_data()};
            int offset_w;
            int fake_h;
            for(int h {0}; h < m_dim[2]; ++h) {
                offset_w = h % jacobi_k;
                fake_h = h / jacobi_k;
                for(int fake_w {0}; fake_w < (m_dim[3] / jacobi_k); ++fake_w) {
                    m_data->at(m_channel_ul.first + h * m_dim[3] + fake_w * jacobi_k + offset_w) = sjfw_data->at(fw_channel_ul.first + fake_h * sjfw_dim[3] + fake_w);
                }
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::zeros(Matrix<MType> &m) {
            matrix_data_p m_data_p = m.get_m_data();
            for(size_t i {0}; i < m_data_p->size(); ++i) {
                m_data_p->at(i) = 0;
            }
            m.resize(m, 0);
        }

        template <typename MType>
        void MakeMatrix<MType>::ones(Matrix<MType>& m) {
            matrix_data_p m_data_p = m.get_m_data();
            for(size_t i {0}; i < m_data_p->size(); ++i) {
                m_data_p->at(i) = 0;
            }
            m.resize(m, 1);
        }
        
        template <typename MType>
        void MakeMatrix<MType>::identity(Matrix<MType>& m) {
            diagonal(m, 1);
        }

        // TODO
        template <typename MType>
        void MakeMatrix<MType>::xavier(Matrix<MType>& m) {
            gaussian(m);
            
        }

        // TODO
        template <typename MType>
        void MakeMatrix<MType>::kaiming(Matrix<MType>& m) {

        }

        template <typename MType>
        void MakeMatrix<MType>::add_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, MType fill_with) {
            // padding[h, w]
            if(padding[0] == 0 && padding[1] == 0) {
                result = m;
                return;
            }
            matrix_dim m_dim {m.get_dim()};
            m_dim[2] += (padding[0] != 0) ? padding[0] * 2 : 0;
            m_dim[3] += (padding[1] != 0) ? padding[1] * 2 : 0;
            result.resize(m_dim, fill_with);
            std::vector<std::thread> thread_c;
            for(int n {0}; n < m_dim[0]; ++n) {
                ul_pos m_batch_ul {m.get_batch(n)};
                ul_pos fw_batch_ul {result.get_batch(n)};
                for(int c {0}; c < m_dim[1]; ++c) {
                    ul_pos m_channel_ul {m.get_channel(c, m_batch_ul)};
                    ul_pos fw_channel_ul {result.get_channel(c, fw_batch_ul)};
                    thread_c.emplace_back(&MakeMatrix<MType>::add_padding_core, this, std::ref(m), std::ref(result), padding, m_channel_ul, fw_channel_ul);
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::add_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned padding, MType fill_with) {
            add_padding(m, result, kernel_shape {padding, padding}, fill_with);
        }

        template <typename MType>
        void MakeMatrix<MType>::add_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned> padding, MType fill_with) {
            add_padding(m, result, kernel_shape {padding}, fill_with);
        }

        template <typename MType>
        void MakeMatrix<MType>::add_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos m_channel_ul, ul_pos fw_channel_ul) {
            // padding[h, w]
            matrix_dim m_dim {m.get_dim()};
            matrix_dim fw_dim {result.get_dim()};
            matrix_data_p m_data {m.get_m_data()};
            matrix_data_p fw_data {result.get_m_data()};
            for(int h {0}; h < m_dim[2]; ++h) {
                for(int w{0}; w < m_dim[3]; ++w) {
                    fw_data->at(fw_channel_ul.first + (padding[0] + h) * fw_dim[3] + padding[1] + w) = m_data->at(m_channel_ul.first + h * m_dim[3] + w);
                }
            }
        }       


        template <typename MType>
        void MakeMatrix<MType>::sub_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding) {
            if(padding[0] == 0 && padding[1] == 0) {
                result = m;
                return;
            }
            matrix_dim m_dim {m.get_dim()};
            m_dim[2] -= (padding[0] != 0) ? padding[0] * 2 : 0;
            m_dim[3] -= (padding[1] != 0) ? padding[1] * 2 : 0;
            assert(m_dim[2] > 0 && m_dim[3] > 0);
            result.resize(m_dim, 0);
            std::vector<std::thread> thread_c;
            for(int n {0}; n < m_dim[0]; ++n) {
                ul_pos m_batch_ul {m.get_batch(n)};
                ul_pos fw_batch_ul {result.get_batch(n)};
                for(int c {0}; c < m_dim[1]; ++c) {
                    ul_pos m_channel_ul {m.get_channel(c, m_batch_ul)};
                    ul_pos fw_channel_ul {result.get_channel(c, fw_batch_ul)};
                    thread_c.emplace_back(&MakeMatrix<MType>::sub_padding_core, this, std::ref(m), std::ref(result), padding, m_channel_ul, fw_channel_ul);
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::sub_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned padding) {
            sub_padding(m, result, kernel_shape {padding, padding});
        }

        template <typename MType>
        void MakeMatrix<MType>::sub_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned> padding) {
            sub_padding(m, result, kernel_shape {padding});
        }

        template <typename MType>
        void MakeMatrix<MType>::sub_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos m_channel_ul, ul_pos fw_channel_ul) {
            matrix_dim m_dim {m.get_dim()};
            matrix_dim fw_dim {result.get_dim()};
            matrix_data_p m_data {m.get_m_data()};
            matrix_data_p fw_data {result.get_m_data()};
            unsigned start_h {padding[0]};
            unsigned start_w {padding[1]};
            for(int h {0}; h < fw_dim[2]; ++h) {
                for(int w {0}; w < fw_dim[3]; ++w) {
                    fw_data->at(fw_channel_ul.first + h * fw_dim[3] + w) = m_data->at(m_channel_ul.first + (start_h + h) * m_dim[3] + start_w + w);
                }
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::img2col(Matrix<MType>& m, Matrix<MType>& fw, kernel_shape kernel_size, unsigned stride) {
            matrix_dim m_dim {m.get_dim()};
            matrix_dim fw_dim {m.get_dim()};
            unsigned output_h {unsigned(std::floor((m_dim[2] - kernel_size[0]) / stride + 1))};
            unsigned output_w {unsigned(std::floor((m_dim[3] - kernel_size[1]) / stride + 1))};
            fw_dim[2] = kernel_size[0] * kernel_size[1];
            fw_dim[3] = output_h * output_w;
            fw.resize(fw_dim, 0);
            std::vector<std::thread> thread_c;
            for(int n {0}; n < m_dim[0]; ++n) {
                ul_pos m_batch_ul {m.get_batch(n)};
                ul_pos fw_batch_ul {fw.get_batch(n)};
                for(int c {0}; c < m_dim[1]; ++c) {
                    ul_pos m_channel_ul {m.get_channel(c, m_batch_ul)};
                    ul_pos fw_channel_ul {fw.get_channel(c, fw_batch_ul)};
                    thread_c.emplace_back(&MakeMatrix<MType>::img2col_core, this, std::ref(m), std::ref(fw), m_channel_ul, fw_channel_ul, kernel_size, stride, output_h, output_w);
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::img2col_core(Matrix<MType>& m, Matrix<MType>& fw, ul_pos m_channel_ul, ul_pos fw_channel_ul, kernel_shape kernel_size, unsigned stride, unsigned output_h, unsigned output_w) {
            matrix_data_p m_data {m.get_m_data()};
            matrix_data_p fw_data {fw.get_m_data()};
            matrix_dim m_dim {m.get_dim()};
            for(unsigned h {0}; h < output_h; ++h) {
                for(unsigned w {0}; w < output_w; ++w) {
                    for(unsigned k_h {0}; k_h < kernel_size[0]; ++k_h) {
                        for(unsigned k_w {0}; k_w < kernel_size[1]; ++k_w) {
                            fw_data->at(fw_channel_ul.first + (k_h * kernel_size[1] + k_w) * (output_h * output_w) + h * output_w + w) = m_data->at(m_channel_ul.first + (h * stride + k_h) * m_dim[3] + w * stride + k_w);
                        }
                    }
                }
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::img2col(Matrix<MType>& m, Matrix<MType>& fw, std::initializer_list<unsigned> kernel_size, unsigned stride) {
            img2col(m, fw, kernel_shape {kernel_size}, stride);
        }

        template <typename MType>
        void MakeMatrix<MType>::img2col(Matrix<MType>& m, Matrix<MType>& fw, unsigned kernel_size, unsigned stride) {
            img2col(m, fw, kernel_shape {kernel_size, kernel_size}, stride);
        }

        template <typename MType>
        void MakeMatrix<MType>::col2img(Matrix<MType>& m, Matrix<MType>& fw, kernel_shape kernel_size, unsigned stride, matrix_dim fw_dim) {
            matrix_dim m_dim {m.get_dim()};
            unsigned output_h {unsigned(std::floor((fw_dim[2] - kernel_size[0]) / stride + 1))};
            unsigned output_w {unsigned(std::floor((fw_dim[3] - kernel_size[1]) / stride + 1))};
            assert(output_h * output_w == m_dim[3]);
            assert(m_dim[0] == fw_dim[0] && m_dim[1] == fw_dim[1]);
            fw.resize(fw_dim, 0);
            std::vector<std::thread> thread_c;
            for(int n {0}; n < m_dim[0]; ++n) {
                ul_pos m_batch_ul {m.get_batch(n)};
                ul_pos fw_batch_ul {fw.get_batch(n)};
                for(int c {0}; c < m_dim[1]; ++c) {
                    ul_pos m_channel_ul {m.get_channel(c, m_batch_ul)};
                    ul_pos fw_channel_ul {fw.get_channel(c, fw_batch_ul)};
                    thread_c.emplace_back(&MakeMatrix<MType>::col2img_core, this, std::ref(m), std::ref(fw), m_channel_ul, fw_channel_ul, kernel_size, stride, output_h, output_w);
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }

        template <typename MType>
        void MakeMatrix<MType>::col2img(Matrix<MType>& m, Matrix<MType>& fw, unsigned kernel_size, unsigned stride, matrix_dim fw_dim) {
            col2img(m, fw, kernel_shape {kernel_size, kernel_size}, stride, fw_dim);
        }

        template <typename MType>
        void MakeMatrix<MType>::col2img(Matrix<MType>& m, Matrix<MType>& fw, std::initializer_list<unsigned> kernel_size, unsigned stride, matrix_dim fw_dim) {
            col2img(m, fw, kernel_shape {kernel_size}, stride, fw_dim);
        }

        template <typename MType>
        // 这里有个问题，例如5x5卷积以 2x2 stride 2卷积过形成的img2col的jacobi，在这里恢复的时候会出错，第五列会全0
        void MakeMatrix<MType>::col2img_core(Matrix<MType>& m, Matrix<MType>& fw, ul_pos m_channel_ul, ul_pos fw_channel_ul, kernel_shape kernel_size, unsigned stride, unsigned output_h, unsigned output_w) {
            matrix_dim m_dim {m.get_dim()};
            matrix_dim fw_dim {fw.get_dim()};
            matrix_data_p m_data {m.get_m_data()};
            matrix_data_p fw_data {fw.get_m_data()};
            for(unsigned h {0}; h < output_h; ++h) {
                for(unsigned w {0}; w < output_w; ++w) {
                    for(unsigned k_h {0}; k_h < kernel_size[0]; ++k_h) {
                        for(unsigned k_w {0}; k_w < kernel_size[1]; ++k_w) {
                            fw_data->at(fw_channel_ul.first + (h * stride + k_h) * fw_dim[2] + w * stride + k_w) += m_data->at(m_channel_ul.first + (k_h * kernel_size[1] + k_w) * m_dim[3] + h * output_h + w);
                        }
                    }
                }
            }
        }
    }
}