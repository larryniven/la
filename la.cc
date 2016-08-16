#include "la/la.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <cblas.h>

namespace la {

    void copy(vector_like<double>& u, vector_like<double> const& v)
    {
        assert(u.size() == v.size());

        std::copy(v.data(), v.data() + v.size(), u.data());
    }

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

    vector<double> sub(vector_like<double> const& u, vector_like<double> const& v)
    {
        vector<double> result { u };

        isub(result, u);

        return result;
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

    void copy(matrix_like<double>& u, matrix_like<double> const& v)
    {
        assert(u.rows() == v.rows() && u.cols() == v.cols());

        std::copy(v.data(), v.data() + v.rows() * v.cols(), u.data());
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
        if (!(u.size() == a.rows() && a.cols() == v.size())) {
            std::cout << "assertion u.size() == a.rows() && a.cols() == v.size() fails" << std::endl;
            std::cout << "u.size() = " << u.size() << std::endl;
            std::cout << "a.rows() = " << a.rows() << std::endl;
            std::cout << "a.cols() = " << a.cols() << std::endl;
            std::cout << "v.size() = " << v.size() << std::endl;
        }

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

    void lmul(vector_like<double>& u,
        vector_like<double> const& v,
        matrix_like<double> const& a)
    {
        if (!(u.size() == a.cols() && a.rows() == v.size())) {
            std::cout << "assertion u.size() == a.cols() && a.rows() == v.size() fails" << std::endl;
            std::cout << "u.size() = " << u.size() << std::endl;
            std::cout << "a.cols() = " << a.cols() << std::endl;
            std::cout << "a.rows() = " << a.rows() << std::endl;
            std::cout << "v.size() = " << v.size() << std::endl;
        }

        cblas_dgemv(CblasRowMajor, CblasTrans, a.rows(), a.cols(), 1, a.data(), a.cols(),
            v.data(), 1, 1, u.data(), 1);
    }

    vector<double> lmul(
        vector_like<double> const& v,
        matrix_like<double> const& a)
    {
        vector<double> result;
        result.resize(a.cols());

        lmul(result, v, a);

        return result;
    }

    void mul(matrix_like<double>& u, matrix_like<double> const& a,
        matrix_like<double> const& b)
    {
        assert(a.cols() == b.rows());
        assert(a.rows() == u.rows() && b.cols() == u.cols());

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, u.rows(), u.cols(), a.cols(),
            1, a.data(), a.cols(), b.data(), b.cols(), 1, u.data(), u.cols());
    }

    matrix<double> mul(matrix_like<double> const& a,
        matrix_like<double> const& b)
    {
        matrix<double> result;
        result.resize(a.rows(), b.cols());
        mul(result, a, b);

        return result;
    }

    double norm(matrix_like<double> const& m)
    {
        return norm(weak_vector<double> { const_cast<double*>(m.data()), m.rows() * m.cols() });
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

    void outer_prod(matrix_like<double>& result,
        vector_like<double> const& a,
        vector_like<double> const& b)
    {
        assert(result.rows() == a.size() && result.cols() == b.size());

        cblas_dger(CblasRowMajor, a.size(), b.size(), 1, a.data(), 1, b.data(), 1,
            result.data(), b.size());
    }

    matrix<double> outer_prod(vector_like<double> const& a,
        vector_like<double> const& b)
    {
        matrix<double> result;
        result.resize(a.size(), b.size());

        outer_prod(result, a, b);

        return result;
    }

    void ltmul(matrix_like<double>& u, matrix_like<double> const& a,
        matrix_like<double> const& b)
    {
        assert(a.rows() == b.rows());
        assert(u.rows() == a.cols() && u.cols() == b.cols());

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, a.cols(), b.cols(), b.rows(),
            1, a.data(), a.cols(), b.data(), b.cols(), 1, u.data(), u.cols());
    }

    void rtmul(matrix_like<double>& u, matrix_like<double> const& a,
        matrix_like<double> const& b)
    {
        assert(a.cols() == b.cols());
        assert(u.rows() == a.rows() && u.cols() == b.rows());

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, a.rows(), b.rows(), a.cols(),
            1, a.data(), a.cols(), b.data(), b.cols(), 1, u.data(), u.cols());
    }

}
