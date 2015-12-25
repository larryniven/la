#ifndef LA_H
#define LA_H

#include <vector>
#include "ebt/ebt.h"

namespace la {

    template <class T>
    struct vector {

        vector();
        explicit vector(std::vector<T> v);
        vector(std::initializer_list<T> const& list);

        T* data();
        T const* data() const;

        unsigned int size() const;

        void resize(unsigned int size, T value = 0);

        T& operator()(int i);
        T const& operator()(int i) const;

        T& at(int i);
        T const& at(int i) const;

    private:
        std::vector<T> vec_;
    };

    template <class T>
    struct matrix;

    template <class T>
    struct matrix {

        matrix();
        explicit matrix(std::vector<std::vector<T>> const& m);
        matrix(std::initializer_list<std::initializer_list<T>> const& list);

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
        std::vector<T> vec_;
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

    double norm(vector<double> const& v);

    double dot(vector<double> const& u, vector<double> const& v);

    void iadd(matrix<double>& u, matrix<double> const& v);
    void isub(matrix<double>& u, matrix<double> const& v);

    vector<double> mult(matrix<double> const& a,
        vector<double> const& v);

    vector<double> lmult(matrix<double> const& a,
        vector<double> const& v);

    template <class T>
    matrix<T> trans(matrix<T> const& m);

    vector<double> tensor_prod(vector<double> const& a,
        vector<double> const& b);

    matrix<double> outer_prod(vector<double> const& a,
        vector<double> const& b);

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
