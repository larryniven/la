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
        struct vector_like {

            virtual ~vector_like();

            virtual T* data() = 0;
            virtual T const* data() const = 0;

            virtual unsigned int size() const = 0;

            virtual T* begin() = 0;
            virtual T const* begin() const = 0;

            virtual T* end() = 0;
            virtual T const* end() const = 0;
        };

        template <class T>
        struct vector : public vector_like<T> {
            vector();
            ~vector();

            vector(vector const& v);
            vector(vector&& v);
            explicit vector(vector_like<T> const& v);
            explicit vector(la::vector_like<T> const& v);

            vector<T>& operator=(vector<T> const& v);
            vector<T>& operator=(vector<T>&& v);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int size() const;

            virtual T* begin();
            virtual T const* begin() const;

            virtual T* end();
            virtual T const* end() const;

            void resize(unsigned int size, T value = 0);

        private:
            T *data_;
            unsigned int size_;
        };

        template <class T>
        la::vector<T> to_host(vector_like<T> const& v);

        template <class T>
        void to_device(vector_like<T>& dv, la::vector_like<T> const& hv);

        template <class T>
        struct weak_vector : public vector_like<T> {

            weak_vector(T *data, unsigned int size);
            weak_vector(vector_like<T>& data);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int size() const;

            virtual T* begin();
            virtual T const* begin() const;

            virtual T* end();
            virtual T const* end() const;

        private:
            T *data_;
            unsigned int size_;
        };

        template <class T>
        struct matrix_like {

            virtual ~matrix_like();

            virtual T* data() = 0;
            virtual T const* data() const = 0;

            virtual unsigned int rows() const = 0;
            virtual unsigned int cols() const = 0;

        };

        template <class T>
        struct matrix : public matrix_like<T> {

            matrix();
            explicit matrix(matrix_like<T> const& m);
            explicit matrix(la::matrix_like<T> const& m);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int rows() const;
            virtual unsigned int cols() const;

            void resize(unsigned int rows, unsigned int cols, T value = 0);

        private:
            vector<T> data_;
            unsigned int rows_;
            unsigned int cols_;
        };

        template <class T>
        la::matrix<T> to_host(matrix_like<T> const& m);

        template <class T>
        void to_device(matrix_like<T>& dm, la::matrix_like<T> const& hm);

        template <class T>
        struct weak_matrix : public matrix_like<T> {

            weak_matrix(matrix_like<T>& m);
            weak_matrix(T *data, unsigned int rows, unsigned int cols);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int rows() const;
            virtual unsigned int cols() const;

        private:
            T *data_;
            unsigned int rows_;
            unsigned int cols_;
        };

        // vector operation

        void copy(vector_like<double>& u, vector_like<double> const& v);

        void zero(vector_like<double>& v);

        void imul(vector_like<double>& u, double d);
        vector<double> mul(vector_like<double> const& u, double d);

        void iadd(vector_like<double>& u, vector_like<double> const& v);
        vector<double> add(vector_like<double> const& u,
            vector_like<double> const& v);

        void isub(vector_like<double>& u, vector_like<double> const& v);
        void idiv(vector_like<double>& u, vector_like<double> const& v);

        void emul(vector_like<double>& z, vector_like<double> const& u,
            vector_like<double> const& v);
        void iemul(vector_like<double>& u, vector_like<double> const& v);
        vector<double> emul(vector_like<double> const& u,
            vector_like<double> const& v);

        double norm(vector_like<double> const& v);

        double dot(vector_like<double> const& u, vector_like<double> const& v);

        // matrix operation

        void copy(matrix_like<double>& u, matrix_like<double> const& v);

        void zero(matrix_like<double>& m);

        void iadd(matrix_like<double>& u, matrix_like<double> const& v);
        void isub(matrix_like<double>& u, matrix_like<double> const& v);

        void mul(vector_like<double>& u, matrix_like<double> const& a,
            vector_like<double> const& v);
        vector<double> mul(matrix_like<double> const& a,
            vector_like<double> const& v);

        vector<double> lmul(matrix_like<double> const& u,
            vector_like<double> const& v);

        double norm(matrix_like<double> const& v);

        vector<double> tensor_prod(vector_like<double> const& a,
            vector_like<double> const& b);

        matrix<double> outer_prod(vector_like<double> const& a,
            vector_like<double> const& b);

        struct idiv_op {
            template <class T>
            __host__ __device__
            void operator()(T t) const;
        };
    }
}

#include "la-gpu-impl.h"

#endif
