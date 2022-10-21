#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <memory>
#include <utility>
#include <cassert>
#include <cmath>
#include <thread>
#include <initializer_list>


namespace aedlf {
    template <typename MType>
    class Matrix{
        public:
            using matrix_data_p = std::shared_ptr<std::vector<MType>>;
            using matrix_data = std::vector<MType>;
            using matrix_dim = std::vector<unsigned long>;
            using slice_parma = std::vector<unsigned long>;
            using ul_pos = const std::pair<unsigned long, unsigned long>;
            Matrix();
            ~Matrix();
            Matrix(matrix_dim shape, MType fill_with);
            Matrix(matrix_dim shape, matrix_data_p data); // 这里仿照caffe使用1d-array
            Matrix(matrix_dim shape, std::initializer_list<MType> init_data);
            Matrix(std::initializer_list<unsigned long> shape, std::initializer_list<MType> init_data);
            Matrix(std::initializer_list<unsigned long> shape, MType fill_with);
            Matrix(std::initializer_list<unsigned long> shape, matrix_data_p data);
            Matrix(const Matrix<MType>& m);
            Matrix<MType>& operator=(const Matrix<MType>& m);
            Matrix<MType>& operator+(const Matrix<MType>& m);
            Matrix<MType>& operator+(MType number);
            Matrix<MType>& operator*(MType scale_number);
            Matrix<MType>& operator*(const Matrix<MType>& m);
            Matrix<MType>& operator+=(const Matrix<MType>& m);
            Matrix<MType>& operator+=(MType number);
            Matrix<MType>& operator*=(const Matrix<MType>& m);
            Matrix<MType>& operator*=(MType number);
            bool operator==(const Matrix<MType>& m);
            void copy_from(const Matrix<MType>& m);
            Matrix<MType>& add(const Matrix<MType>& addend); // inplace计算
            Matrix<MType>& add(MType number);
            Matrix<MType>& mul(const Matrix<MType>& mutiplier); // inplace计算
            Matrix<MType>& mul_v(const Matrix<MType>& mutiplier); // inplace计算
            Matrix<MType>& scale(MType scale_number);
            void clear_data();
            void set_data(matrix_data_p data);
            void resize(matrix_dim shape, MType fill_with);
            void resize(unsigned long n, unsigned long c, unsigned long h, unsigned long w, MType fill_with);
            void view(matrix_dim shape);
            void view(unsigned long n, unsigned long c, unsigned long h, unsigned long w);
            void view(std::initializer_list<unsigned long> shape);
            void concat(Matrix<MType>& m, unsigned long dim);
            Matrix<MType> sum_by_dim(unsigned long sum_dim);
            Matrix<MType> slice(std::initializer_list<unsigned long> slice_index);
            Matrix<MType> slice(slice_parma slice_index);
            Matrix<MType> slice(unsigned long slice_dim, unsigned long start, unsigned long end);
            Matrix<MType> slice(unsigned long slice_dim, std::initializer_list<unsigned long> range);
            void T();
            bool is_uninitialized();
            void check_initialized() const;
            MType get(unsigned long n, unsigned long c, unsigned long h, unsigned long w);
            MType get(unsigned long index);
            void set(unsigned long n, unsigned long c, unsigned long h, unsigned long w, MType value);
            void set(unsigned long index, MType value);
            ul_pos get_batch(matrix_dim dim, unsigned long batch_id) const;
            ul_pos get_batch(unsigned long batch_id) const;
            ul_pos get_channel(matrix_dim dim, unsigned long channel_id, ul_pos batch_pos) const;
            ul_pos get_channel(unsigned long channel_id, ul_pos batch_pos) const;
            const matrix_data_p get_data() const;
            const matrix_dim get_dim() const;
            matrix_data_p get_m_data();
        private:
            void mul_core(const matrix_data_p inplace_m, ul_pos inplace_m_pos, matrix_dim inplace_m_dim, const matrix_data_p mul_m, ul_pos mul_m_pos, matrix_dim mul_m_dim, matrix_data_p result, ul_pos result_pos);
            void add_boardcast_core(Matrix<MType>& summand, const Matrix<MType>& addend, ul_pos channel_ul, ul_pos a_channel_ul, int piece);
            void mul_v_boardcast_core(Matrix<MType>& multiplied, const Matrix<MType>& mutiplier, ul_pos channel_ul, ul_pos m_channel_ul, int piece);
            void T_core(ul_pos channel_ul);
            unsigned long T_core_get_next(unsigned long now_index, unsigned long m_h, unsigned long m_len);
            void sum_by_dim_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long sum_dim, unsigned long batch_id);
            void concat_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long dim, unsigned long batch_id);
            matrix_data_p data;
            matrix_dim shape; // (n,c,h,w)
            bool uninitialized {false};
    };

    template <typename MType>
    Matrix<MType>::Matrix() {
        uninitialized = true;
        data = std::make_shared<matrix_data>(0, MType(0));
        shape = matrix_dim {0,0,0,0};
    }

    template <typename MType>
    Matrix<MType>::Matrix(matrix_dim shape, std::initializer_list<MType> init_data) {
        assert(shape[0] * shape[1] * shape[2] * shape[3] == init_data.size()); // warning here
        this->shape = shape;
        data = std::make_shared<matrix_data>(init_data);
    }

    template <typename MType>
    Matrix<MType>::Matrix(std::initializer_list<unsigned long> shape, std::initializer_list<MType> init_data) {
        assert(shape.size() == 4);
        new (this)Matrix(matrix_dim {shape}, init_data);
    }

    template <typename MType>
    Matrix<MType>::Matrix(std::initializer_list<unsigned long> shape, MType fill_with) {
        assert(shape.size() == 4);
        new (this)Matrix(matrix_dim {shape}, fill_with);
    }

    template <typename MType>
    Matrix<MType>::~Matrix() {
        data.reset();
    }

    template <typename MType>
    Matrix<MType>::Matrix(matrix_dim shape, MType fill_with) {
        assert(shape.size() == 4);
        long data_len {1};
        for(int dim : shape) {
            assert(dim >= 1);
            data_len *= dim;
        }
        data = std::make_shared<matrix_data>(data_len, fill_with);
        this->shape = shape;
    }

    template <typename MType>
    Matrix<MType>::Matrix(matrix_dim shape, matrix_data_p data) {
        assert(shape.size() == 4);
        long data_len {1};
        for(int dim : shape) {
            assert(dim >= 1);
            data_len *= dim;
        }
        assert(data->size() == data_len);
        this->data = data;
        this->shape = shape;
    }

    template <typename MType>
    Matrix<MType>::Matrix(std::initializer_list<unsigned long> shape, matrix_data_p data) {
        assert(shape.size() == 4);
        new (this)Matrix(matrix_dim {shape}, data);
    }

    template <typename MType>
    Matrix<MType>::Matrix(const Matrix<MType>& m) {
        data = m.data;
        shape = m.shape;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator=(const Matrix<MType>& m) {
        if(uninitialized) {
            this->data = m.data;
            this->shape = m.shape;
            uninitialized = false;
            return *this;
        }
        if(*this == m) {
            return *this;
        }
        else {
            this->data = m.data;
            this->shape = m.shape;
        }
        uninitialized = false;
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator+(const Matrix<MType>& m) {
        add(m);
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator+(MType number) {
        add(number);
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator*(const Matrix<MType>& m) {
        check_initialized();
        mul(m);
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator*(MType scale_number) {
        scale(scale_number);
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator+=(const Matrix<MType>& m) {
        add(m);
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator+=(MType number) {
        add(number);
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator*=(const Matrix<MType>& m) {
        mul(m);
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::operator*=(MType number) {
        scale(number);
        return *this;
    }

    template <typename MType>
    bool Matrix<MType>::operator==(const Matrix<MType>& m) {
        check_initialized();
        if(this->data != m.data) {
            return false;
        }
        if(this->shape != m.shape) {
            return false;
        }
        return true;
    }

    template <typename MType>
    void Matrix<MType>::copy_from(const Matrix<MType>& m) {
        shape = m.shape;
        data->resize(shape[0] * shape[1] * shape[2] * shape[3], MType(0));
        for(size_t i {0}; i < data->size(); ++i) {
            data->at(i) = m.data->at(i);
        }
        uninitialized = false;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::add(const Matrix<MType>& addend) {
        check_initialized();
        assert(addend.shape[0] == shape[0] && addend.shape[1] == shape[1]);
        if(this->data->size() == (addend.data)->size()) {
            for(size_t i {0}; i < data->size(); ++i) {
                data->at(i) += (addend.data)->at(i);
            }
        }
        else if(this->data->size() % (addend.data)->size() == 0 && this->data->size() >= (addend.data)->size()) {
            std::vector<std::thread> thread_c;
            unsigned long max_piece = data->size() / addend.data->size();
            for(unsigned long n {0}; n < shape[0]; ++n) {
                ul_pos this_batch_ul {get_batch(n)};
                ul_pos addend_batch_ul {addend.get_batch(n)};
                for(unsigned long c {0}; c < shape[1]; ++c) {
                    ul_pos this_channel_ul {get_channel(c, this_batch_ul)};
                    ul_pos addend_channel_ul {addend.get_channel(c, addend_batch_ul)};
                    for(unsigned long piece {0}; piece < max_piece; ++piece) {
                        thread_c.emplace_back(&Matrix<MType>::add_boardcast_core, this, std::ref(*this), addend, this_channel_ul, addend_channel_ul, piece);
                    }
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }
        else if((addend.data)->size() % this->data->size() == 0 && (addend.data)->size() >= this->data->size()) {
            Matrix<MType> bigger_one {addend};
            std::vector<std::thread> thread_c;
            size_t max_piece = data->size() / addend.data->size();
            for(unsigned long n {0}; n < shape[0]; ++n) {
                ul_pos this_batch_ul {get_batch(n)};
                ul_pos addend_batch_ul {addend.get_batch(n)};
                for(unsigned long c {0}; c < shape[1]; ++c) {
                    ul_pos this_channel_ul {get_channel(c, this_batch_ul)};
                    ul_pos addend_channel_ul {addend.get_channel(c, addend_batch_ul)};
                    for(unsigned long piece {0}; piece < max_piece; ++piece) {
                        thread_c.emplace_back(&Matrix<MType>::add_boardcast_core, this, std::ref(bigger_one), *this, this_channel_ul, addend_channel_ul, piece);
                    }
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
            data = bigger_one.data;
            shape = bigger_one.shape;
        }
        else {
            throw std::runtime_error("Matrix is shape is not match");
        }
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::add(MType number) {
        check_initialized();
        for(size_t i {0}; i < data->size(); ++i) {
            data->at(i) += number;
        }
    }

    template <typename MType>
    void Matrix<MType>::add_boardcast_core(Matrix<MType>& summand, const Matrix<MType>& addend, ul_pos channel_ul, ul_pos a_channel_ul, int piece) {
        assert(summand.data->size() > addend.data->size());
        assert(summand.data->size() % addend.data->size() == 0);
        size_t max_piece = summand.data->size() / addend.data->size();
        assert(piece < max_piece);
        matrix_dim addend_dim {addend.get_dim()};
        assert(summand.shape[2] % addend_dim[2] == 0);
        assert(summand.shape[3] % addend_dim[3] == 0);
        unsigned long piece_w, piece_x, piece_y;
        piece_w = summand.shape[3] / addend_dim[3];
        piece_x = piece % piece_w; //x is w
        piece_y = piece / piece_w; // y is h
        for(unsigned long addend_h {0}; addend_h < addend_dim[2]; ++addend_h) {
            for(unsigned long addend_w {0}; addend_w < addend_dim[3]; ++addend_w) {
                summand.data->at(channel_ul.first + (piece_y * addend_dim[2] + addend_h) * summand.shape[2] + piece_x * addend_dim[3] + addend_w) += addend.data->at(a_channel_ul.first + addend_h * addend_dim[2] + addend_w);
            }
        }
    }

    template <typename MType>
    MType Matrix<MType>::get(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {
        check_initialized();
        assert(n >= 0 && n < shape[0]);
        assert(c >= 0 && c < shape[1]);
        assert(h >= 0 && h < shape[2]);
        assert(w >= 0 && w < shape[3]);
        return data->at(
            n * shape[1] * shape[2] * shape[3] + c * shape[2] * shape[3] + h * shape[3] + w
        );
    }

    template <typename MType>
    MType Matrix<MType>::get(unsigned long index) {
        assert(index >= 0 && index < data->size());
        return data->at(index);
    }

    template <typename MType>
    void Matrix<MType>::set(unsigned long n, unsigned long c, unsigned long h, unsigned long w, MType value) {
        check_initialized();
        assert(n >= 0 && n < shape[0]);
        assert(c >= 0 && c < shape[1]);
        assert(h >= 0 && h < shape[2]);
        assert(w >= 0 && w < shape[3]);
        data->at(
            n * shape[1] * shape[2] * shape[3] + c * shape[2] * shape[3] + h * shape[3] + w
        ) = value;
    }

    template <typename MType>
    void Matrix<MType>::set(unsigned long index, MType value) {
        check_initialized();
        assert(index < data->size());
        data->at(index) = value;
    }

    template <typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_batch(matrix_dim dim, unsigned long batch_id) const {
        check_initialized();
        assert(batch_id < dim[0]);
        unsigned long batch_len {dim[1] * dim[2] * dim[3]};
        std::pair<int, int> pos {batch_id * batch_len, (batch_id + 1) * batch_len - 1};
        // return std::as_const(pos); // C++17
        return static_cast<const ul_pos>(pos); // C++11
    }

    template <typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_batch(unsigned long batch_id) const {
        check_initialized();
        return get_batch(shape, batch_id);
    }

    template <typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_channel(matrix_dim dim, unsigned long channel_id, ul_pos batch_pos) const {
        check_initialized();
        assert(channel_id < dim[1]);
        unsigned long channel_len {dim[2] * dim[3]};
        std::pair<int, int> pos {channel_id * channel_len + batch_pos.first, (channel_id + 1) * channel_len + batch_pos.first - 1};
        assert (pos.second <= batch_pos.second);
        // return std::as_const(pos);
        return static_cast<const ul_pos>(pos); // C++11
    }

    template <typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_channel(unsigned long channel_id, ul_pos batch_pos) const {
        check_initialized();
        return get_channel(shape, channel_id, batch_pos);
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::scale(MType scale_number) {
        check_initialized();
        for(size_t i {0}; i < data->size(); ++i) {
            data->at(i) *= scale_number;
        }
        return *this;
    }

    template <typename MType>
    void Matrix<MType>::mul_core(const matrix_data_p inplace_m, ul_pos inplace_m_pos, matrix_dim inplace_m_dim, const matrix_data_p mul_m, ul_pos mul_m_pos, matrix_dim mul_m_dim, matrix_data_p result, ul_pos result_pos) {
        // (h, w) -> implace_m_dim (w, h) -> mul_m_dim
        assert(inplace_m_dim.size() == 2 && mul_m_dim.size() == 2);
        assert(inplace_m_dim[0] == mul_m_dim[1] && inplace_m_dim[1] == mul_m_dim[0]);
        assert(result->size() % inplace_m_dim[0] * mul_m_dim[1] == 0);
        unsigned long result_h, result_w, i;
        MType sum;
        for(result_h = 0; result_h < inplace_m_dim[0]; ++result_h) {
            for(result_w = 0; result_w < mul_m_dim[1]; ++result_w) {
                sum = MType(0);
                for(i = 0; i < inplace_m_dim[1]; ++i) {
                    sum += inplace_m->at(inplace_m_pos.first + result_h * inplace_m_dim[1] + i) * mul_m->at(mul_m_pos.first + i * inplace_m_dim[0] + result_w);
                }
                result->at(result_pos.first + result_h * mul_m_dim[1] + result_w) = sum;
            }
        }
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::mul(const Matrix<MType>& mutiplier) {
        check_initialized();
        assert(shape[0] == mutiplier.shape[0] && shape[1] == mutiplier.shape[1]);
        matrix_data_p mul_result {std::make_shared<matrix_data>()};
        mul_result->resize(shape[0] * shape[1] * shape[2] * mutiplier.shape[3], 0);
        matrix_dim mul_dim {shape[0], shape[1], shape[2], mutiplier.shape[3]};
        std::vector<std::thread> thread_c;
        for(unsigned long n {0}; n < shape[0]; ++n) {
            ul_pos this_batch_ul {get_batch(shape, n)};
            ul_pos m_batch_ul {get_batch(mutiplier.shape, n)};
            for(unsigned long c {0}; c < shape[1]; ++c) {
                ul_pos this_channel_ul {get_channel(shape, c, this_batch_ul)};
                ul_pos m_channel_ul {get_channel(mutiplier.shape, c, m_batch_ul)};
                matrix_dim this_dim {shape[2], shape[3]};
                matrix_dim m_dim {mutiplier.shape[2], mutiplier.shape[3]};
                ul_pos result_batch_ul = get_batch(mul_dim, n);
                ul_pos result_channel_ul = get_channel(mul_dim, c, result_batch_ul);
                // thread_c.push_back(std::thread {&Matrix<MType>::mul_core, this, std::as_const(data), this_channel_ul, this_dim, std::as_const(mutiplier.data), m_channel_ul, m_dim, mul_result, result_channel_ul});
                thread_c.push_back(std::thread {&Matrix<MType>::mul_core, this, static_cast<const matrix_data_p>(data), this_channel_ul, this_dim, static_cast<const matrix_data_p>(mutiplier.data), m_channel_ul, m_dim, mul_result, result_channel_ul});
            }
        }
        for(auto& t : thread_c) {
            t.join();
        }
        data.reset();
        data = mul_result;
        shape = mul_dim;
        return *this;
    }

    template <typename MType>
    Matrix<MType>& Matrix<MType>::mul_v(const Matrix<MType> &mutiplier) {
        check_initialized();
        if(data->size() == mutiplier.data->size()) {
            for(unsigned long i {0}; i < shape[0]; ++i) {
                data->at(i) *= mutiplier.data->at(i);
            }
        }
        else if(data->size() % mutiplier.data->size() == 0 && this->data->size() >= mutiplier.data->size()) {
            std::vector<std::thread> thread_c;
            size_t max_piece = data->size() / mutiplier.data->size();
            for(unsigned long n {0}; n < shape[0]; ++n) {
                ul_pos this_batch_ul {get_batch(n)};
                ul_pos addend_batch_ul {mutiplier.get_batch(n)};
                for(unsigned long c {0}; c < shape[1]; ++c) {
                    ul_pos this_channel_ul {get_channel(c, this_batch_ul)};
                    ul_pos addend_channel_ul {mutiplier.get_channel(c, addend_batch_ul)};
                    for(unsigned long piece {0}; piece < max_piece; ++piece) {
                        thread_c.emplace_back(&Matrix<MType>::mul_v_boardcast_core, this, std::ref(*this), mutiplier, this_channel_ul, addend_channel_ul, piece);
                    }
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
        }
        else if(mutiplier.data->size() % data->size() == 0 && mutiplier.data->size() >= this->data->size()) {
            Matrix<MType> bigger_one {mutiplier};
            std::vector<std::thread> thread_c;
            size_t max_piece = data->size() / mutiplier.data->size();
            for(unsigned long n {0}; n < shape[0]; ++n) {
                ul_pos this_batch_ul {get_batch(n)};
                ul_pos addend_batch_ul {mutiplier.get_batch(n)};
                for(unsigned long c {0}; c < shape[1]; ++c) {
                    ul_pos this_channel_ul {get_channel(c, this_batch_ul)};
                    ul_pos addend_channel_ul {mutiplier.get_channel(c, addend_batch_ul)};
                    for(unsigned long piece {0}; piece < max_piece; ++piece) {
                        thread_c.emplace_back(&Matrix<MType>::mul_v_boardcast_core, this, std::ref(bigger_one), *this, this_channel_ul, addend_channel_ul, piece);
                    }
                }
            }
            for(auto& t : thread_c) {
                t.join();
            }
            data = bigger_one.data;
            shape = bigger_one.shape;
        }
        else {
            throw std::runtime_error("Matrix shape is not match");
        }
        return *this;
    }

    template <typename MType>
    void Matrix<MType>::mul_v_boardcast_core(Matrix<MType>& multiplied, const Matrix<MType>& mutiplier, ul_pos channel_ul, ul_pos m_channel_ul, int piece) {
        assert(multiplied.data->size() > mutiplier.data->size());
        assert(multiplied.data->size() % mutiplier.data->size() == 0);
        size_t max_piece = multiplied.data->size() / mutiplier.data->size();
        assert(piece < max_piece);
        matrix_dim mutiplier_dim {mutiplier.get_dim()};
        assert(multiplied.shape[2] % mutiplier_dim[2] == 0);
        assert(multiplied.shape[3] % mutiplier_dim[3] == 0);
        unsigned long piece_w, piece_x, piece_y;
        piece_w = multiplied.shape[3] / mutiplier_dim[3];
        piece_x = piece % piece_w; //x is w
        piece_y = piece / piece_w; // y is h
        for(unsigned long addend_h {0}; addend_h < mutiplier_dim[2]; ++addend_h) {
            for(unsigned long addend_w {0}; addend_w < mutiplier_dim[3]; ++addend_w) {
                multiplied.data->at(channel_ul.first + (piece_y * mutiplier_dim[2] + addend_h) * multiplied.shape[2] + piece_x * mutiplier_dim[3] + addend_w) *= mutiplier.data->at(m_channel_ul.first + addend_h * mutiplier_dim[2] + addend_w);
            }
        }
    }

    template <typename MType>
    void Matrix<MType>::clear_data() {
        uninitialized = true;
        data->resize(0, 0);
        shape = matrix_dim {0,0,0,0};
    }

    template <typename MType>
    void Matrix<MType>::set_data(matrix_data_p data) {
        if(this->data == data) {
            return;
        }
        this->data = data;
        uninitialized = false;
    }

    template <typename MType>
    void Matrix<MType>::resize(matrix_dim shape, MType fill_with) {
        assert(shape.size() == 4);
        assert(shape[0] >= 0 && shape[1] >= 0 && shape[2] >= 0 && shape[3] >= 0);
        if(this->shape == shape) {
            return;
        }
        this->shape = shape;
        data->resize(shape[0] * shape[1] * shape[2] * shape[3], fill_with);
        uninitialized = false;
    }

    template <typename MType>
    void Matrix<MType>::resize(unsigned long n, unsigned long c, unsigned long h, unsigned long w, MType fill_with) {
        assert(n >= 1);
        assert(c >= 1);
        assert(h >= 1);
        assert(w >= 1);
        matrix_dim new_dim {n, c, h, w};
        resize(new_dim, fill_with);
    }

    template <typename MType>
    void Matrix<MType>::view(matrix_dim shape) {
        check_initialized();
        assert(shape[0] * shape[1] * shape[2] * shape[3] == this->shape[0] * this->shape[1] * this->shape[2] * this->shape[3]);
        this->shape = shape;
    }

    template <typename MType>
    void Matrix<MType>::view(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {
        view(matrix_dim {n, c, h, w});
    }

    template <typename MType>
    void Matrix<MType>::view(std::initializer_list<unsigned long> shape) {
        view(matrix_dim {shape});
    }

    template <typename MType>
    Matrix<MType> Matrix<MType>::sum_by_dim(unsigned long sum_dim) {
        check_initialized();
        if(sum_dim == -1) {
            matrix_dim new_shape {1,1,1,1};
            MType sum_num {0};
            for(size_t i {0}; i < data->size(); ++i) {
                sum_num += data->at(i);
            }
            return Matrix<MType> {new_shape, sum_num};
        }
        else {
            assert(sum_dim >=0 && sum_dim < 4);
            matrix_dim new_shape {shape};
            new_shape[sum_dim] = 1;
            Matrix<MType> new_matrix {new_shape, 0};
            for(unsigned long n {0}; n < shape[0]; ++n) {
                sum_by_dim_core(*this, new_matrix, sum_dim, n);
            }
            return new_matrix;
        }
    }

    template <typename MType>
    void Matrix<MType>::sum_by_dim_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long sum_dim, unsigned long batch_id) {
        if(sum_dim == 1) {
            // channel sum
            for(unsigned long h {0}; h < m.shape[2]; ++h) {
                for(unsigned long w {0}; w < m.shape[3]; ++w) {
                    MType sum_value {0};
                    for(unsigned long c {0}; c < m.shape[1]; ++c) {
                        sum_value += m.get(batch_id, c, h, w);
                    }
                    result.set(batch_id, 0, h, w, sum_value);
                }
            }
        }
        else if(sum_dim == 2) {
            // height sum
            for(unsigned long c {0}; c < m.shape[1]; ++c) {
                for(unsigned long w {0}; w < m.shape[3]; ++w) {
                    MType sum_value {0};
                    for(unsigned long h {0}; h < m.shape[2]; ++h) {
                        sum_value += m.get(batch_id, c, h, w);
                    }
                    result.set(batch_id, c, 0, w, sum_value);
                }
            }
        }
        else if(sum_dim == 3) {
            // width sum
            for(unsigned long c {0}; c < m.shape[1]; ++c) {
                for(unsigned long h {0}; h < m.shape[2]; ++h) {
                    MType sum_value {0};
                    for(unsigned long w {0}; w < m.shape[3]; ++w) {
                        sum_value += m.get(batch_id, c, h, w);
                    }
                    result.set(batch_id, c, h, 0, sum_value);
                }
            }
        }
        else {
            return;
        }
    }

    template <typename MType>
    void Matrix<MType>::T() {
        check_initialized();
        std::vector<std::thread> thread_c;
        for(unsigned long n {0}; n < shape[0]; ++n) {
            ul_pos batch_ul {get_batch(n)};
            for(unsigned long c{0}; c < shape[1]; ++c) {
                ul_pos channel_ul {get_channel(c, batch_ul)};
                thread_c.emplace_back(&Matrix<MType>::T_core, this, channel_ul);
            }
        }
        for(auto& t : thread_c) {
            t.join();
        }
        resize(shape[0], shape[1], shape[3], shape[2], MType(0));
    }

    template <typename MType>
    void Matrix<MType>::T_core(ul_pos channel_ul) {
        // 这里转置转换的位置关系是：index -> (index * h) % (len - 1)，有环，需要思考怎么o(1)空间处理环
        unsigned long m_len {shape[2] * shape[3]};
        for(unsigned long i {0}; i < m_len; ++i) {
            unsigned long next_index = T_core_get_next(i, shape[2], m_len);
            while(i < next_index) {
                next_index = T_core_get_next(next_index, shape[2], m_len);
            }
            if(i == next_index) {               
                unsigned long next = T_core_get_next(i, shape[2], m_len);
                MType pre = data->at(channel_ul.first + i);
                MType temp;
                while(i != next) {
                    temp = data->at(channel_ul.first + next);
                    data->at(channel_ul.first + next) = pre;
                    pre = temp;
                    next = T_core_get_next(next, shape[2], m_len);
                }
                data->at(channel_ul.first + next) = pre;
            }
        }
    }

    template <typename MType>
    unsigned long Matrix<MType>::T_core_get_next(unsigned long now_index, unsigned long m_h, unsigned long m_len) {
        return (now_index * m_h) % (m_len - 1);
    }

    template <typename MType>
    bool Matrix<MType>::is_uninitialized() {
        return uninitialized;
    }

    template <typename MType>
    void Matrix<MType>::concat(Matrix<MType> &m, unsigned long dim) {
        check_initialized();
        assert(dim > 0 && dim <= 3);
        matrix_dim result_dim {shape};
        result_dim[dim] += m.shape[dim];
        Matrix<MType> result {result_dim, 0};
        std::vector<std::thread> thread_c;
        for(unsigned long n {0}; n < shape[0]; ++n) {
            thread_c.emplace_back(&Matrix<MType>::concat_core, this, std::ref(m), std::ref(result), dim, n);
        }
        for(auto& t : thread_c) {
            t.join();
        }
        this->copy_from(result);
    }

    template <typename MType>
    void Matrix<MType>::concat_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long dim, unsigned long batch_id) {
        if(dim == 1) {
            assert(this->shape[0] == m.shape[0]);
            assert(this->shape[2] == m.shape[2]);
            assert(this->shape[3] == m.shape[3]);
            matrix_dim result_dim {result.shape};
            for(unsigned long inner_c {0}; inner_c < shape[1]; ++inner_c) {
                for(unsigned long h {0}; h < shape[2]; ++h) {
                    for(unsigned long w {0}; w < shape[3]; ++w) {
                        result.set(batch_id, inner_c, h, w, this->get(batch_id, inner_c, h, w));
                    }
                }
            }
            for(unsigned long inner_c {0}; inner_c < m.shape[1]; ++inner_c) {
                for(unsigned long h {0}; h < shape[2]; ++h) {
                    for(unsigned long w {0}; w < shape[3]; ++w) {
                        result.set(batch_id, inner_c + shape[1], h, w, m.get(batch_id, inner_c, h, w));
                    }
                }
            }
        }
        else if(dim == 2) {
            assert(this->shape[0] == m.shape[0]);
            assert(this->shape[1] == m.shape[1]);
            assert(this->shape[3] == m.shape[3]);
            matrix_dim result_dim {result.shape};
            for(unsigned long c {0}; c < shape[1]; ++c) {
                for(unsigned long inner_h {0}; inner_h < shape[2]; ++inner_h) {
                    for(unsigned long w {0}; w < shape[3]; ++w) {
                        result.set(batch_id, c, inner_h, w, this->get(batch_id, c, inner_h, w));
                    }
                }
                for(unsigned long inner_h {0}; inner_h < m.shape[2]; ++inner_h) {
                    for(unsigned long w {0}; w < shape[3]; ++w) {
                        result.set(batch_id, c, inner_h + shape[2], w, m.get(batch_id, c, inner_h, w));
                    }
                }
            }
        }
        else if(dim == 3) {
            assert(this->shape[0] == m.shape[0]);
            assert(this->shape[1] == m.shape[1]);
            assert(this->shape[2] == m.shape[2]);
            matrix_dim result_dim {result.shape};
            for(unsigned long c {0}; c < shape[1]; ++c) {
                for(unsigned long h {0}; h < shape[2]; ++h) {
                    for(unsigned long inner_w {0}; inner_w < shape[3]; ++inner_w) {
                        result.set(batch_id, c, h, inner_w, this->get(batch_id, c, h, inner_w));
                    }
                    for(unsigned long inner_w {0}; inner_w < m.shape[3]; ++inner_w) {
                        result.set(batch_id, c, h, inner_w + shape[3], m.get(batch_id, c, h, inner_w));
                    }
                }
            }
        }
    }

    template <typename MType>
    Matrix<MType> Matrix<MType>::slice(unsigned long slice_dim, unsigned long start, unsigned long end) {
        check_initialized();
        slice_parma sr_dim {0,0,0,0,0,0,0,0};
        sr_dim[slice_dim * 2] = start;
        sr_dim[slice_dim * 2 + 1] = end;
        for(size_t i {0}; i < 8; i+=2) {
            if(i == slice_dim) {
                continue;
            }
            sr_dim[i * 2 + 1] = shape[i / 2];
        }
        return slice(sr_dim);
    }

    template <typename MType>
    Matrix<MType> Matrix<MType>::slice(unsigned long slice_dim, std::initializer_list<unsigned long> range) {
        check_initialized();
        assert(range.size() == 2);
        slice_parma sr_dim {0,0,0,0,0,0,0,0};
        sr_dim[slice_dim * 2] = *range.begin();
        sr_dim[slice_dim * 2 + 1] = *(range.begin() + 1);
        for(size_t i {0}; i < 8; i+=2) {
            if(i / 2 == slice_dim) {
                continue;
            }
            sr_dim[i + 1] = shape[i / 2];
        }
        return slice(sr_dim);
    }

    template <typename MType>
    Matrix<MType> Matrix<MType>::slice(std::initializer_list<unsigned long> slice_index) {
        return slice(slice_parma {slice_index});
    }

    template <typename MType>
    Matrix<MType> Matrix<MType>::slice(slice_parma slice_index) {
        check_initialized();
        // slice_index {n[start], n[end], c[start], c[end], h[start], h[end], w[start], w[end]}
        assert(slice_index.size() == 8);
        for(size_t slice_i {0}; slice_i < slice_index.size(); ++slice_i) {
            if(slice_index[slice_i] < 0) {
                slice_index[slice_i] += shape[slice_i / 2];
            }
            if(slice_index[slice_i] > shape[slice_i / 2]) {
                throw std::runtime_error("Slice index out of range");
            }
        }
        matrix_dim sr_dim {0,0,0,0};
        for(size_t slice_i {0}; slice_i < slice_index.size(); slice_i+=2) {
            if(slice_index[slice_i + 1] < slice_index[slice_i]) {
                throw std::runtime_error("Slice range invalid");
            }
            sr_dim[slice_i / 2] = slice_index[slice_i + 1] - slice_index[slice_i];
        }
        Matrix<MType> slice_result {sr_dim, 0};
        int sw_index {0};
        for(unsigned long n {slice_index[0]}; n < slice_index[1]; ++n) {
            for(unsigned long c {slice_index[2]}; c < slice_index[3]; ++c) {
                for(unsigned long h {slice_index[4]}; h < slice_index[5]; ++h) {
                    for(unsigned long w {slice_index[6]}; w < slice_index[7]; ++w) {
                        slice_result.set(sw_index++, this->get(n, c, h, w));
                    }
                }
            }
        }
        return slice_result;
    }

    template <typename MType>
    inline void Matrix<MType>::check_initialized() const {
        if(uninitialized) {
            throw std::runtime_error("Matrix is uninitialized, please initialized first");
        }
    }

    // 后面两个函数存在bug，实际上并不能阻止修改shared_ptr的内容, 要用const T&修改
    template <typename MType>
    const typename Matrix<MType>::matrix_data_p Matrix<MType>::get_data() const {
        check_initialized();
        // return std::as_const(data);
        return static_cast<const matrix_data_p>(data);
    }

    template <typename MType>
    const typename Matrix<MType>::matrix_dim Matrix<MType>::get_dim() const {
        check_initialized();
        // return std::as_const(shape);
        return static_cast<const matrix_dim>(shape);
    }

    template <typename MType>
    typename Matrix<MType>::matrix_data_p Matrix<MType>::get_m_data() {
        check_initialized();
        return data;
    }
};