#ifndef LA_GPU_H
#define LA_GPU_H

#include "la/la.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cassert>
#include <thrust/tuple.h>
#include "ebt/ebt.h"

namespace la {

    namespace gpu {

        __global__ void print_vec(double const *p, int size);
        __global__ void print_mat(double const *p, int rows, int cols);

        struct device {
            static device d;

            device();
            ~device();

            cublasHandle_t handle;

            static device& get_instance();
            static cublasHandle_t& get_handle();
        };

        template <class T>
        struct vector_view {
            vector_view(T const* data, int size);

            T const* data() const;

            unsigned int size() const;

            T const& operator()(int i) const;

            T const& at(int i) const;

            T const* begin();

            T const* end() const;

        private:
            T const* data_;
            unsigned int size_;
        };

        template <class T>
        struct vector {

            vector();
            explicit vector(la::vector<T> const& v);
            vector(vector<T>&& v);
            vector(vector<T> const& v);

            vector<T>& operator=(vector<T> const& v);
            vector<T>& operator=(vector<T>&& v);

            ~vector();

            T* data();
            T const* data() const;

            unsigned int size() const;

            void resize(unsigned int size, T value = 0);

            T& operator()(int i);
            T const& operator()(int i) const;

            T& at(int i);
            T const& at(int i) const;

            T* begin();
            T const* begin() const;

            T* end();
            T const* end() const;

        private:
            T* data_;
            unsigned int size_;
        };

        template <class T>
        struct matrix {

            matrix();
            explicit matrix(la::matrix<T> const& m);

            T* data();
            T const* data() const;

            unsigned int rows() const;

            unsigned int cols() const;

            void resize(int rows, int cols, T value = 0);

            T& operator()(unsigned int r, unsigned int c);
            T const& operator()(unsigned int r, unsigned int c) const;

            T& at(unsigned int r, unsigned int c);
            T const& at(unsigned int r, unsigned int c) const;

        private:
            vector<T> vec_;
            unsigned int rows_;
            unsigned int cols_;
        };

        // vector operation

        template <class T>
        la::vector<T> to_host(vector<T> const& v);

        template <class T>
        void to_device(vector<T>& dv, la::vector<T> const& hv);

        void zero(vector<double>& v);

        void imul(vector<double>& u, double d);
        vector<double> mul(vector<double> u, double d);

        void iadd(vector<double>& u, vector<double> const& v);
        vector<double> add(vector<double> u,
            vector<double> const& v);

        void isub(vector<double>& u, vector<double> const& v);
        void idiv(vector<double>& u, vector<double> const& v);

        void emul(vector<double>& z, vector<double> const& u,
            vector<double> const& v);
        void iemul(vector<double>& u, vector<double> const& v);
        vector<double> emul(vector<double> u,
            vector<double> const& v);

        double norm(std::vector<double> const& v);

        double dot(vector<double> const& u, vector<double> const& v);

        // matrix operation

        template <class T>
        la::matrix<T> to_host(matrix<T> const& m);

        template <class T>
        void to_device(matrix<T>& dm, la::matrix<T> const& hm);

        void zero(matrix<double>& m);

        void iadd(matrix<double>& u, matrix<double> const& v);
        void isub(matrix<double>& u, matrix<double> const& v);

        void mul(vector<double>& u, matrix<double> const& a,
            vector<double> const& v);
        vector<double> mul(matrix<double> const& a,
            vector<double> const& v);

        vector<double> lmul(matrix<double> const& u,
            vector<double> const& v);

        vector<double> tensor_prod(vector<double> const& a,
            vector<double> const& b);

        matrix<double> outer_prod(vector<double> const& a,
            vector<double> const& b);

        struct idiv_op {
            template <class T>
            __host__ __device__
            void operator()(T t) const;
        };

    }
}

#include "la-gpu-impl.h"

#endif
