#include "la/la.h"
#include <cblas.h>
#include <cmath>
#include <cassert>

namespace la {

    void imul(vector<double>& u, double d)
    {
        cblas_dscal(u.size(), d, u.data(), 1);
    }

    void iadd(vector<double>& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u(i) += v(i);
        }
    }

    void isub(vector<double>& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u(i) -= v(i);
        }
    }

    void imul(vector<double>& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u(i) *= v(i);
        }
    }

    void idiv(vector<double>& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u(i) /= v(i);
        }
    }

    vector<double> add(
        vector<double> const& u,
        vector<double> const& v)
    {
        assert(u.size() == v.size());

        vector<double> result;
        result.resize(u.size());

        for (int i = 0; i < u.size(); ++i) {
            result(i) = u(i) + v(i);
        }

        return result;
    }

    double norm(vector<double> const& v)
    {
        double sum = 0;

        for (int i = 0; i < v.size(); ++i) {
            sum += v(i) * v(i);
        }

        return std::sqrt(sum);
    }

    double dot(vector<double> const& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        double sum = 0;
        for (int i = 0; i < u.size(); ++i) {
            sum += u(i) * v(i);
        }
        return sum;
    }

    vector<double> logistic(vector<double> const& v)
    {
        vector<double> result;
        result.resize(v.size());

        for (int i = 0; i < v.size(); ++i) {
            result(i) = 1 / (1 + std::exp(-v(i)));
        }

        return result;
    }

    void iadd(matrix<double>& u, matrix<double> const& v)
    {
        assert(u.rows() == v.rows());
        assert(u.cols() == v.cols());

        for (int i = 0; i < u.rows(); ++i) {
            for (int j = 0; j < u.cols(); ++j) {
                u(i, j) += v(i, j);
            }
        }
    }

    void isub(matrix<double>& u, matrix<double> const& v)
    {
        assert(u.rows() == v.rows());
        assert(u.cols() == v.cols());

        for (int i = 0; i < u.rows(); ++i) {
            for (int j = 0; j < u.cols(); ++j) {
                u(i, j) -= v(i, j);
            }
        }
    }

    vector<double> mult(
        matrix<double> const& u,
        vector<double> const& v)
    {
        vector<double> result;
        result.resize(u.rows());

        cblas_dgemv(CblasRowMajor, CblasNoTrans, u.rows(), u.cols(), 1, u.data(), u.cols(),
            v.data(), 1, 1, result.data(), 1);

        return result;
    }

    vector<double> dyadic_prod(vector<double> const& a,
        vector<double> const& b)
    {
        vector<double> result;

        result.resize(a.size() * b.size());

        for (int i = 0; i < a.size(); ++i) {
            for (int j = 0; j < b.size(); ++j) {
                result(i * a.size() + j) = a(i) * b(j);
            }
        }

        return result;
    }
}
