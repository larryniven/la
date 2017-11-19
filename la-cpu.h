#ifndef LA_CPU_H
#define LA_CPU_H

#include "la/la.h"
#include <vector>
#include "ebt/ebt.h"
#include <cassert>

namespace la {

    namespace cpu {

        template <class T>
        struct vector_like
            : public la::vector_like<T> {

            virtual T& operator()(int i) = 0;
            virtual T const& operator()(int i) const = 0;

            virtual T& at(int i) = 0;
            virtual T const& at(int i) const = 0;
        };

        template <class T>
        struct vector : public vector_like<T> {

            vector();
            explicit vector(std::vector<T> const& data);
            explicit vector(std::vector<T>&& data);
            explicit vector(vector_like<T> const& v);
            vector(std::initializer_list<T> list);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int size() const;

            virtual T& operator()(int i);
            virtual T const& operator()(int i) const;

            virtual T& at(int i);
            virtual T const& at(int i) const;

            void resize(unsigned int size, T value = 0);

        private:
            std::vector<T> data_;

        };

        template <class T>
        struct weak_vector : public vector_like<T> {

            weak_vector(vector_like<T>& data);
            weak_vector(T *data, unsigned int size);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int size() const;

            virtual T& operator()(int i);
            virtual T const& operator()(int i) const;

            virtual T& at(int i);
            virtual T const& at(int i) const;

        private:
            T *data_;
            unsigned int size_;
        };

        template <class T>
        struct matrix_like
            : public la::matrix_like<T> {

            virtual T& operator()(unsigned int r, unsigned int c) = 0;
            virtual T const& operator()(unsigned int r, unsigned int c) const = 0;

            virtual T& at(unsigned int r, unsigned int c) = 0;
            virtual T const& at(unsigned int r, unsigned int c) const = 0;

            virtual vector_like<T>& as_vector() = 0;
            virtual vector_like<T> const& as_vector() const = 0;
        };

        template <class T>
        struct matrix : public matrix_like<T> {

            matrix();
            explicit matrix(std::vector<std::vector<T>> data);
            explicit matrix(matrix_like<T> const& m);
            matrix(std::initializer_list<std::initializer_list<T>> list);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int rows() const;
            virtual unsigned int cols() const;

            virtual T& operator()(unsigned int r, unsigned int c);
            virtual T const& operator()(unsigned int r, unsigned int c) const;

            virtual T& at(unsigned int r, unsigned int c);
            virtual T const& at(unsigned int r, unsigned int c) const;

            void resize(unsigned int rows, unsigned int cols, T value = 0);

            virtual vector_like<T>& as_vector();
            virtual vector_like<T> const& as_vector() const;

        private:
            vector<T> data_;
            unsigned int rows_;
            unsigned int cols_;
        };

        template <class T>
        struct weak_matrix : public matrix_like<T> {

            weak_matrix(matrix_like<T>& m);
            weak_matrix(T *data, unsigned int rows, unsigned int cols);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int rows() const;
            virtual unsigned int cols() const;

            virtual T& operator()(unsigned int r, unsigned int c);
            virtual T const& operator()(unsigned int r, unsigned int c) const;

            virtual T& at(unsigned int r, unsigned int c);
            virtual T const& at(unsigned int r, unsigned int c) const;

            virtual vector_like<T>& as_vector();
            virtual vector_like<T> const& as_vector() const;

        private:
            weak_vector<T> data_;
            unsigned int rows_;
            unsigned int cols_;
        };

        template <class T>
        struct tensor_like
            : public la::tensor_like<T> {

            virtual T& operator()(std::vector<int> indices) = 0;
            virtual T const& operator()(std::vector<int> indices) const = 0;

            virtual T& at(std::vector<int> indices) = 0;
            virtual T const& at(std::vector<int> indices) const = 0;

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
            tensor(tensor_like<T> const& that);
            tensor(vector<T>&& data, std::vector<unsigned int> sizes);
            tensor(vector_like<T> const& data, std::vector<unsigned int> sizes);
            explicit tensor(vector_like<T> const& v);
            explicit tensor(matrix_like<T> const& m);

            virtual T* data();
            virtual T const* data() const;

            virtual unsigned int dim() const;
            virtual unsigned int size(unsigned int d) const;

            virtual T& operator()(std::vector<int> indices);
            virtual T const& operator()(std::vector<int> indices) const;

            virtual T& at(std::vector<int> indices);
            virtual T const& at(std::vector<int> indices) const;

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

            weak_matrix<double> mat_;
        };

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

            virtual T& operator()(std::vector<int> indices);
            virtual T const& operator()(std::vector<int> indices) const;

            virtual T& at(std::vector<int> indices);
            virtual T const& at(std::vector<int> indices) const;

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

        void axpy(vector_like<double>& y, double a, vector_like<double> const& x);

        void emul(vector_like<double>& z, vector_like<double> const& u,
            vector_like<double> const& v);

        void div(vector_like<double>& u, double d, vector_like<double> const& v);

        void ediv(vector_like<double>& z, vector_like<double> const& u,
            vector_like<double> const& v);

        double norm(vector_like<double> const& v);

        double dot(vector_like<double> const& u, vector_like<double> const& v);

        bool has_nan(vector_like<double> const& u);

        // matrix operation

        void copy(matrix_like<double>& u, matrix_like<double> const& v);

        void zero(matrix_like<double>& m);

        void axpy(matrix_like<double>& y, double a, matrix_like<double> const& x);

        void emul(vector_like<double>& z, vector_like<double> const& u,
            vector_like<double> const& v);

        void div(matrix_like<double>& z, double d, matrix_like<double>& u);

        void ediv(matrix_like<double>& z, matrix_like<double> const& u,
            matrix_like<double> const& v);

        void mul(vector_like<double>& u, matrix_like<double> const& a,
            vector_like<double> const& v);

        void lmul(vector_like<double>& u, 
            vector_like<double> const& v, matrix_like<double> const& a);

        void mul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b);

        void ltmul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b);

        void rtmul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b);

        void vdot(vector_like<double>& v, matrix_like<double> const& a,
            matrix_like<double> const& b);

        double norm(matrix_like<double> const& m);

        void outer_prod(matrix_like<double>& result,
            vector_like<double> const& a,
            vector_like<double> const& b);

        bool has_nan(matrix_like<double> const& m);

        template <class T>
        matrix<T> trans(matrix_like<T> const& m);

        // tensor operation

        void copy(tensor_like<double>& u, tensor_like<double> const& v);

        void zero(tensor_like<double>& m);

        void axpy(tensor_like<double>& y, double a, tensor_like<double> const& x);

        void emul(tensor_like<double>& z, tensor_like<double> const& u,
            tensor_like<double> const& v);

        void div(tensor_like<double>& z, double d, tensor_like<double>& u);

        void ediv(tensor_like<double>& z, tensor_like<double> const& u,
            tensor_like<double> const& v);

        void mul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& v);

        void ltmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b);

        void rtmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b);

        void resize_as(tensor<double>& a, tensor_like<double> const& b, double value = 0);

        double dot(tensor_like<double> const& a, tensor_like<double> const& b);

        void vdot(tensor_like<double>& v, tensor_like<double> const& a,
            tensor_like<double> const& b);

        double norm(tensor_like<double> const& v);

        bool has_nan(tensor_like<double> const& m);

        /*
         * 
         * f1: first dimension of the filters
         * f2: second dimension of the filters
         * 
         * p1: first dimension of padding
         * p2: second dimension of padding
         * 
         * d1: first dimension of dilation
         * d2: second dimension of dilation
         *
         * this function ignores the dimension of result.
         * 
         * u is assumed to have dimension batch size x time x freq x channel
         * 
         */
        void corr_linearize(tensor_like<double>& result,
            tensor_like<double> const& u, int f1, int f2,
            int p1, int p2, int d1, int d2);

        void corr_linearize(tensor_like<double>& result,
            tensor_like<double> const& u, int f1, int f2);

        /*
         * 
         * f1: first dimension of the filters
         * f2: second dimension of the filters
         * 
         * p1: first dimension of padding
         * p2: second dimension of padding
         * 
         * d1: first dimension of dilation
         * d2: second dimension of dilation
         *
         * this function ignores the dimension of u.
         * 
         * result is assumed to have dimension batch size x time x freq x channel
         * 
         */
        void corr_delinearize(tensor_like<double>& result,
            tensor_like<double> const& u,
            int f1, int f2, int p1, int p2, int d1, int d2);

    }

}

namespace ebt {
    namespace json {
    
        template <class T>
        struct json_parser<la::cpu::vector<T>> {
            la::cpu::vector<T> parse(std::istream& is);
        };
    
        template <class T>
        struct json_parser<la::cpu::matrix<T>> {
            la::cpu::matrix<T> parse(std::istream& is);
        };
    
        template <class T>
        struct json_parser<la::cpu::tensor<T>> {
            la::cpu::tensor<T> parse(std::istream& is);
        };
    
        template <class T>
        struct json_writer<la::cpu::vector<T>> {
            void write(la::cpu::vector<T> const& v, std::ostream& os);
        };
        
        template <class T>
        struct json_writer<la::cpu::matrix<T>> {
            void write(la::cpu::matrix<T> const& m, std::ostream& os);
        };

        template <class T>
        struct json_writer<la::cpu::tensor<T>> {
            void write(la::cpu::tensor<T> const& m, std::ostream& os);
        };
    
    }
}

#include "la-cpu-impl.h"

#endif
