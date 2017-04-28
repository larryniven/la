#ifndef LA_GPU_H
#define LA_GPU_H

#include "la/la.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cassert>
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

            vector<T>& operator=(vector_like<T> const& v);
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

            virtual vector_like<T>& as_vector() = 0;
            virtual vector_like<T> const& as_vector() const = 0;
        };

        template <class T>
        struct matrix : public matrix_like<T> {

            matrix();
            matrix(matrix<T> const& that);
            matrix(matrix<T>&& that);
            explicit matrix(matrix_like<T> const& m);
            explicit matrix(la::matrix_like<T> const& m);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int rows() const;
            virtual unsigned int cols() const;

            void resize(unsigned int rows, unsigned int cols, T value = 0);

            virtual vector_like<T>& as_vector();
            virtual vector_like<T> const& as_vector() const;

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

            virtual vector_like<T>& as_vector();
            virtual vector_like<T> const& as_vector() const;

        private:
            weak_vector<T> data_;
            unsigned int rows_;
            unsigned int cols_;
        };

        // tensor

        template <class T>
        struct tensor_like {
            virtual ~tensor_like();

            virtual T* data() = 0;
            virtual T const* data() const = 0;

            virtual unsigned int dim() const = 0;
            virtual unsigned int size(unsigned int d) const = 0;

            virtual unsigned int vec_size() const = 0;
            virtual std::vector<unsigned int> sizes() const = 0;

            virtual vector_like<T>& as_vector() = 0;
            virtual vector_like<T> const& as_vector() const = 0;

            virtual matrix_like<T>& as_matrix() = 0;
            virtual matrix_like<T> const& as_matrix() const = 0;

        };

        template <class T>
        struct tensor
            : public tensor_like<T> {

            tensor();
            tensor(tensor<T>&& that);
            tensor(tensor<T> const& that);
            tensor(la::tensor_like<T> const& ht);
            tensor(vector<T>&& data, std::vector<unsigned int> sizes);
            tensor(vector_like<T> const& data, std::vector<unsigned int> sizes);
            explicit tensor(vector_like<T> const& v);
            explicit tensor(matrix_like<T> const& m);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int dim() const;
            virtual unsigned int size(unsigned int d) const;

            void resize(std::vector<unsigned int> sizes, T value = 0);

            tensor<T>& operator=(tensor<T>&& that);
            tensor<T>& operator=(tensor<T> const& that);

            virtual unsigned int vec_size() const;
            virtual std::vector<unsigned int> sizes() const;

            virtual vector_like<T>& as_vector();
            virtual vector_like<T> const& as_vector() const;

            virtual matrix_like<T>& as_matrix();
            virtual matrix_like<T> const& as_matrix() const;

        private:
            vector<T> data_;
            std::vector<unsigned int> sizes_;
            unsigned int dim_;
            unsigned int vec_size_;

            weak_vector<double> vec_;
            weak_matrix<double> mat_;
        };

        template <class T>
        la::tensor<T> to_host(tensor_like<T> const& t);

        template <class T>
        void to_device(tensor_like<T>& dt, la::tensor_like<T> const& ht);

        template <class T>
        struct weak_tensor
            : public tensor_like<T> {

            weak_tensor(tensor_like<T>& t);
            weak_tensor(T *data, std::vector<unsigned int> sizes);
            explicit weak_tensor(vector_like<T> const& v);
            explicit weak_tensor(matrix_like<T> const& m);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int dim() const;
            virtual unsigned int size(unsigned int d) const;

            virtual unsigned int vec_size() const;
            virtual std::vector<unsigned int> sizes() const;

            virtual vector_like<T>& as_vector();
            virtual vector_like<T> const& as_vector() const;

            virtual matrix_like<T>& as_matrix();
            virtual matrix_like<T> const& as_matrix() const;

        private:
            weak_vector<double> data_;
            std::vector<unsigned int> sizes_;
            unsigned int dim_;

            weak_matrix<double> mat_;
        };

        // vector operation

        void copy(vector_like<double>& u, vector_like<double> const& v);

        void zero(vector_like<double>& v);

        void imul(vector_like<double>& u, double d);
        vector<double> mul(vector<double>&& u, double d);
        vector<double> mul(vector_like<double> const& u, double d);

        void iadd(vector_like<double>& u, vector_like<double> const& v);
        vector<double> add(vector_like<double> const& u,
            vector_like<double> const& v);
        vector<double> add(vector<double>&& u,
            vector_like<double> const& v);
        vector<double> add(vector_like<double> const& u,
            vector<double>&& v);

        void isub(vector_like<double>& u, vector_like<double> const& v);
        vector<double> sub(vector_like<double> const& u, vector_like<double> const& v);

        void idiv(vector_like<double>& u, vector_like<double> const& v);

        void emul(vector_like<double>& z, vector_like<double> const& u,
            vector_like<double> const& v);
        void iemul(vector_like<double>& u, vector_like<double> const& v);
        vector<double> emul(vector_like<double> const& u,
            vector_like<double> const& v);

        double norm(vector_like<double> const& v);

        double dot(vector_like<double> const& u, vector_like<double> const& v);

        bool has_nan(vector_like<double> const& u);

        void axpy(vector_like<double>& y, double a, vector_like<double> const& x);

        // matrix operation

        void copy(matrix_like<double>& u, matrix_like<double> const& v);

        void zero(matrix_like<double>& m);

        void imul(matrix_like<double>& u, double d);
        matrix<double> mul(matrix<double>&& u, double d);
        matrix<double> mul(matrix_like<double> const& u, double d);

        void iadd(matrix_like<double>& u, matrix_like<double> const& v);
        void isub(matrix_like<double>& u, matrix_like<double> const& v);

        void mul(vector_like<double>& u, matrix_like<double> const& a,
            vector_like<double> const& v);
        vector<double> mul(matrix_like<double> const& a,
            vector_like<double> const& v);

        void lmul(vector_like<double>& u, 
            vector_like<double> const& v, matrix_like<double> const& a);
        vector<double> lmul(vector_like<double> const& v,
            matrix_like<double> const& a);

        void mul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b);
        matrix<double> mul(matrix_like<double> const& a,
            matrix_like<double> const& b);

        void ltmul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b);
        void rtmul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b);

        double norm(matrix_like<double> const& v);

        vector<double> tensor_prod(vector_like<double> const& a,
            vector_like<double> const& b);

        void outer_prod(matrix_like<double>& result,
            vector_like<double> const& a,
            vector_like<double> const& b);

        matrix<double> outer_prod(vector_like<double> const& a,
            vector_like<double> const& b);

        bool has_nan(matrix_like<double> const& m);

        void axpy(matrix_like<double>& y, double a, matrix_like<double> const& x);

        // tensor operation

        void copy(tensor_like<double>& u, tensor_like<double> const& v);

        void zero(tensor_like<double>& m);

        void imul(tensor_like<double>& u, double a);

        void mul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& v);
        void ltmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b);
        void rtmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b);

        tensor<double> mul(tensor_like<double> const& m,
            double a);
        tensor<double> mul(tensor_like<double> const& a,
            tensor_like<double> const& v);

        void resize_as(tensor<double>& a, tensor_like<double> const& b, double value = 0);

        void emul(tensor_like<double>& z, tensor_like<double> const& u,
            tensor_like<double> const& v);

        void iadd(tensor_like<double>& a, tensor_like<double> const& b);

        void isub(tensor_like<double>& a, tensor_like<double> const& b);

        double dot(tensor_like<double> const& a, tensor_like<double> const& b);

        double norm(tensor_like<double> const& v);

        bool has_nan(tensor_like<double> const& m);

        void axpy(tensor_like<double>& y, double a, tensor_like<double> const& x);

    }
}

#include "la-gpu-impl.h"

#endif
