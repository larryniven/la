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

        cblas_daxpy(u.size(), 1, v.data(), 1, u.data(), 1);
    }

    void isub(vector<double>& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        cblas_daxpy(u.size(), -1, v.data(), 1, u.data(), 1);
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

    vector<double> add(vector<double> u, vector<double> const& v)
    {
        iadd(u, v);
        return u;
    }

    vector<double> mult(vector<double> u, double d)
    {
        imul(u, d);
        return u;
    }

    double norm(vector<double> const& v)
    {
        return cblas_dnrm2(v.size(), v.data(), 1);
    }

    double dot(vector<double> const& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        return cblas_ddot(u.size(), u.data(), 1, v.data(), 1);
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

    vector<double> lmult(
        matrix<double> const& u,
        vector<double> const& v)
    {
        vector<double> result;
        result.resize(u.rows());

        cblas_dgemv(CblasRowMajor, CblasTrans, u.rows(), u.cols(), 1, u.data(), u.cols(),
            v.data(), 1, 1, result.data(), 1);

        return result;
    }

    vector<double> tensor_prod(vector<double> const& a,
        vector<double> const& b)
    {
        vector<double> result;
        result.resize(a.size() * b.size());

        cblas_dger(CblasRowMajor, a.size(), b.size(), 1, a.data(), 1, b.data(), 1,
            result.data(), b.size());

        return result;
    }

    matrix<double> outer_prod(vector<double> const& a,
        vector<double> const& b)
    {
        matrix<double> result;
        result.resize(a.size(), b.size());

        cblas_dger(CblasRowMajor, a.size(), b.size(), 1, a.data(), 1, b.data(), 1,
            result.data(), b.size());

        return result;
    }
}
