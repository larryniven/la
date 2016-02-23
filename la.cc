#include "la/la.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <cblas.h>

namespace la {

    void zero(vector_like<double>& u)
    {
        std::memset(u.data(), 0, u.size() * sizeof(double));
    }

    void imul(vector_like<double>& u, double d)
    {
        cblas_dscal(u.size(), d, u.data(), 1);
    }

    vector<double> mul(vector<double>&& u, double d)
    {
        vector<double> result { std::move(u) };

        imul(result, d);

        return result;
    }

    vector<double> mul(vector_like<double> const& u, double d)
    {
        vector<double> result { u };

        imul(result, d);

        return result;
    }

    void iadd(vector_like<double>& u, vector_like<double> const& v)
    {
        assert(u.size() == v.size());

        cblas_daxpy(u.size(), 1, v.data(), 1, u.data(), 1);
    }

    vector<double> add(vector_like<double> const& u, vector_like<double> const& v)
    {
        vector<double> result { u };

        iadd(result, v);

        return result;
    }

    vector<double> add(vector<double>&& u,
        vector_like<double> const& v)
    {
        vector<double> result { std::move(u) };

        iadd(result, v);

        return result;
    }

    vector<double> add(vector_like<double> const& u,
        vector<double>&& v)
    {
        vector<double> result { std::move(v) };

        iadd(result, u);

        return result;
    }

    void isub(vector_like<double>& u, vector_like<double> const& v)
    {
        assert(u.size() == v.size());

        cblas_daxpy(u.size(), -1, v.data(), 1, u.data(), 1);
    }

    void idiv(vector_like<double>& u, vector_like<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u(i) /= v(i);
        }
    }

    void emul(vector_like<double>& z,
        vector_like<double> const& u,
        vector_like<double> const& v)
    {
        assert(u.size() == v.size() && z.size() == v.size());

        cblas_dgbmv(CblasRowMajor, CblasNoTrans, u.size(), u.size(), 0, 0,
            1.0, u.data(), 1, v.data(), 1, 1.0, z.data(), 1);
    }

    void iemul(vector_like<double>& u, vector_like<double> const& v)
    {
        emul(u, u, v);
    }

    vector<double> emul(
        vector_like<double> const& u,
        vector_like<double> const& v)
    {
        vector<double> result;
        result.resize(u.size());

        emul(result, u, v);

        return result;
    }

    double norm(vector_like<double> const& v)
    {
        return cblas_dnrm2(v.size(), v.data(), 1);
    }

    double dot(vector_like<double> const& u, vector_like<double> const& v)
    {
        assert(u.size() == v.size());

        return cblas_ddot(u.size(), u.data(), 1, v.data(), 1);
    }

    void zero(matrix_like<double>& m)
    {
        std::memset(m.data(), 0, m.rows() * m.cols() * sizeof(double));
    }

    void imul(matrix_like<double>& u, double d)
    {
        weak_vector<double> v(u.data(), u.rows() * u.cols());
        imul(v, d);
    }

    matrix<double> mul(matrix<double>&& u, double d)
    {
        matrix<double> result { std::move(u) };

        imul(result, d);

        return result;
    }

    matrix<double> mul(matrix_like<double> const& u, double d)
    {
        matrix<double> result { u };

        imul(result, d);

        return result;
    }

    void iadd(matrix_like<double>& u, matrix_like<double> const& v)
    {
        assert(u.rows() == v.rows());
        assert(u.cols() == v.cols());

        la::weak_vector<double> a {u.data(), u.rows() * u.cols()};
        la::weak_vector<double> b {const_cast<double*>(v.data()), v.rows() * v.cols()};

        iadd(a, b);
    }

    void isub(matrix_like<double>& u, matrix_like<double> const& v)
    {
        assert(u.rows() == v.rows());
        assert(u.cols() == v.cols());

        la::weak_vector<double> a {u.data(), u.rows() * u.cols()};
        la::weak_vector<double> b {const_cast<double*>(v.data()), v.rows() * v.cols()};

        isub(a, b);
    }

    void mul(vector_like<double>& u, matrix_like<double> const& a,
        vector_like<double> const& v)
    {
        assert(u.size() == a.rows() && a.cols() == v.size());

        cblas_dgemv(CblasRowMajor, CblasNoTrans, a.rows(), a.cols(), 1, a.data(), a.cols(),
            v.data(), 1, 1, u.data(), 1);
    }

    vector<double> mul(
        matrix_like<double> const& u,
        vector_like<double> const& v)
    {
        vector<double> result;
        result.resize(u.rows());

        mul(result, u, v);

        return result;
    }

    vector<double> lmul(
        matrix_like<double> const& u,
        vector_like<double> const& v)
    {
        vector<double> result;
        result.resize(u.rows());

        cblas_dgemv(CblasRowMajor, CblasTrans, u.rows(), u.cols(), 1, u.data(), u.cols(),
            v.data(), 1, 1, result.data(), 1);

        return result;
    }

    vector<double> tensor_prod(vector_like<double> const& a,
        vector_like<double> const& b)
    {
        vector<double> result;
        result.resize(a.size() * b.size());

        cblas_dger(CblasRowMajor, a.size(), b.size(), 1, a.data(), 1, b.data(), 1,
            result.data(), b.size());

        return result;
    }

    matrix<double> outer_prod(vector_like<double> const& a,
        vector_like<double> const& b)
    {
        matrix<double> result;
        result.resize(a.size(), b.size());

        cblas_dger(CblasRowMajor, a.size(), b.size(), 1, a.data(), 1, b.data(), 1,
            result.data(), b.size());

        return result;
    }
}
