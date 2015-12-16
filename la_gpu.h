#ifndef LA_GPU_H
#define LA_GPU_H

#include <cublas_v2.h>
#include <vector>
#include <cassert>

namespace la_gpu {

    struct device {
        static device d;

        device();
        ~device();

        cublasHandle_t handle;

        static device& get_instance();
        static cublasHandle_t& get_handle();
    };

    template <class T>
    struct vector {

        vector()
            : data_(nullptr)
        {}

        vector(vector<T>&& v)
        {
             data_ = v.data_;
             v.data_ = nullptr;
             size_ = v.size_;
        }

        vector(vector<T> const& v)
        {
            cudaMalloc(&data_, sizeof(T) * v.size());
            cudaMemcpy(data_, v.data(), sizeof(T) * v.size(), cudaMemcpyDeviceToDevice);
            size_ = v.size();
        }

        vector<T>& operator=(vector<T> const& v)
        {
            cudaFree(data_);
            cudaMalloc(&data_, sizeof(T) * v.size());
            cudaMemcpy(data_, v.data(), sizeof(T) * v.size(), cudaMemcpyDeviceToDevice);
            size_ = v.size();
            return *this;
        }

        vector<T>& operator=(vector<T>&& v)
        {
            data_ = v.data_;
            v.data_ = nullptr;
            size_ = v.size_;

            return *this;
        }

        ~vector()
        {
            cudaFree(data_);
        }

        T* data()
        {
            return data_;
        }

        T const* data() const
        {
            return data_;
        }

        unsigned int size() const
        {
            return size_;
        }

        void resize(unsigned int size, T value = 0)
        {
            T* result;
            cudaMalloc(&result, sizeof(T) * size);
            cudaMemset(result, value, size_);
            cudaMemcpy(result, data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice);
            cudaFree(data_);
            data_ = result;
            size_ = size;
        }

        T& operator()(int i)
        {
            return data_[i];
        }

        T const& operator()(int i) const
        {
            return data_[i];
        }

        T& at(int i)
        {
            assert(i < size_);
            return data_[i];
        }

        T const& at(int i) const
        {
            assert(i < size_);
            return data_[i];
        }

    private:
        T* data_;
        unsigned int size_;
    };

    template <class T>
    struct matrix {

        T* data()
        {
            return vec_.data();
        }

        T const* data() const
        {
            return vec_.data();
        }

        unsigned int rows() const
        {
            return rows_;
        }

        unsigned int cols() const
        {
            return cols_;
        }

        void resize(int rows, int cols, T value = 0)
        {
            vec_.resize(cols * rows, value);
            rows_ = rows;
            cols_ = cols;
        }

        T& operator()(unsigned int r, unsigned int c)
        {
            return vec_(c * rows_ + r);
        }

        T const& operator()(unsigned int r, unsigned int c) const
        {
            return vec_(c * rows_ + r);
        }

        T& at(unsigned int r, unsigned int c)
        {
            return vec_.at(c * rows_ + r);
        }

        T const& at(unsigned int r, unsigned int c) const
        {
            return vec_.at(c * rows_ + r);
        }

    private:
        vector<T> vec_;
        unsigned int rows_;
        unsigned int cols_;
    };

    void imul(vector<double>& u, double d);

    void iadd(vector<double>& u, vector<double> const& v);
    void isub(vector<double>& u, vector<double> const& v);
    void imul(vector<double>& u, vector<double> const& v);
    void idiv(vector<double>& u, vector<double> const& v);

    vector<double> add(vector<double> u,
        vector<double> const& v);

    double norm(std::vector<double> const& v);

    double dot(vector<double> const& u, vector<double> const& v);

    vector<double> logistic(vector<double> const& v);

    void iadd(matrix<double>& u, matrix<double> const& v);
    void isub(matrix<double>& u, matrix<double> const& v);

    vector<double> mult(matrix<double> const& a,
        vector<double> const& v);

    vector<double> lmult(matrix<double> const& a,
        vector<double> const& v);

    vector<double> dyadic_prod(vector<double> const& a,
        vector<double> const& b);
}

#endif
