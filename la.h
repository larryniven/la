#ifndef LA_H
#define LA_H

#include <vector>

namespace la {

    template <class T>
    struct vector;

    template <class T>
    vector<T> to_vector(std::vector<T> v);

    template <class T>
    struct vector {

        vector()
        {}

        vector(std::initializer_list<T> const& list)
            : vec_(list)
        {}

        T* data()
        {
            return vec_.data();
        }

        T const* data() const
        {
            return vec_.data();
        }

        unsigned int size() const
        {
            return vec_.size();
        }

        void resize(unsigned int size, T value = 0)
        {
            return vec_.resize(size, value);
        }

        T& operator()(int i)
        {
            return vec_[i];
        }

        T const& operator()(int i) const
        {
            return vec_[i];
        }

        T& at(int i)
        {
            return vec_.at(i);
        }

        T const& at(int i) const
        {
            return vec_.at(i);
        }

        friend vector to_vector<T>(std::vector<T> v);

    private:
        std::vector<T> vec_;

    };

    template <class T>
    vector<T> to_vector(std::vector<T> v)
    {
        vector<T> result;
        result.vec_ = std::move(v);
        return result;
    }

    template <class T>
    struct matrix {

        matrix()
        {}

        matrix(std::initializer_list<std::initializer_list<T>> const& list)
        {
            rows_ = list.size();
            for (auto& ell: list) {
                cols_ = ell.size();
                vec_.insert(vec_.end(), ell.begin(), ell.end());
            }
        }

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
            vec_.resize(rows * cols, value);
            rows_ = rows;
            cols_ = cols;
        }

        T& operator()(unsigned int r, unsigned int c)
        {
            return vec_[r * cols_ + c];
        }

        T const& operator()(unsigned int r, unsigned int c) const
        {
            return vec_[r * cols_ + c];
        }

        T& at(unsigned int r, unsigned int c)
        {
            return vec_.at(r * cols_ + c);
        }

        T const& at(unsigned int r, unsigned int c) const
        {
            return vec_.at(r * cols_ + c);
        }

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
