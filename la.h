#ifndef LA_H
#define LA_H

#include <vector>
#include "ebt/ebt.h"
#include <cassert>

namespace la {

    template <class T>
    struct vector_like {

        virtual ~vector_like();

        virtual T* data() = 0;
        virtual T const* data() const = 0;

        virtual unsigned int size() const = 0;

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
    struct matrix_like {

        virtual ~matrix_like();

        virtual T* data() = 0;
        virtual T const* data() const = 0;

        virtual unsigned int rows() const = 0;
        virtual unsigned int cols() const = 0;

        virtual T& operator()(unsigned int r, unsigned int c) = 0;
        virtual T const& operator()(unsigned int r, unsigned int c) const = 0;

        virtual T& at(unsigned int r, unsigned int c) = 0;
        virtual T const& at(unsigned int r, unsigned int c) const = 0;

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

    private:
        T *data_;
        unsigned int rows_;
        unsigned int cols_;
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

    double norm(matrix_like<double> const& m);

    template <class T>
    matrix<T> trans(matrix_like<T> const& m);

    vector<double> tensor_prod(vector_like<double> const& a,
        vector_like<double> const& b);

    void outer_prod(matrix_like<double>& result,
        vector_like<double> const& a,
        vector_like<double> const& b);

    matrix<double> outer_prod(vector_like<double> const& a,
        vector_like<double> const& b);
}

namespace ebt {
    namespace json {
    
        template <class T>
        struct json_parser<la::vector<T>> {
            la::vector<T> parse(std::istream& is);
        };
    
        template <class T>
        struct json_parser<la::matrix<T>> {
            la::matrix<T> parse(std::istream& is);
        };
    
        template <class T>
        struct json_writer<la::vector<T>> {
            void write(la::vector<T> const& v, std::ostream& os);
        };
        
        template <class T>
        struct json_writer<la::matrix<T>> {
            void write(la::matrix<T> const& m, std::ostream& os);
        };
    
    }
}

#include "la-impl.h"

#endif
