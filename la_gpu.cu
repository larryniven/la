#include "la/la_gpu.h"
#include <cmath>
#include <cassert>
#include <cublas_v2.h>
#include <cblas.h>

namespace la_gpu {

    device device::d = device();

    device::device()
    {
        cublasCreate(&handle);
    }

    device::~device()
    {
        cublasDestroy(handle);
    }

    device& device::get_instance()
    {
        return d;
    }

    cublasHandle_t& device::get_handle()
    {
        return get_instance().handle;
    }

    void imul(vector<double>& u, double d)
    {
        cublasDscal(device::get_handle(), u.size(), &d, u.data(), 1);
    }

    void iadd(vector<double>& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        double alpha = 1;
        cublasDaxpy(device::get_handle(), u.size(), &alpha, v.data(), 1, u.data(), 1);
    }

    void isub(vector<double>& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        double alpha = -1;
        cublasDaxpy(device::get_handle(), u.size(), &alpha, v.data(), 1, u.data(), 1);
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
        vector<double> u,
        vector<double> const& v)
    {
        iadd(u, v);
        return u;
    }

    double norm(vector<double> const& v)
    {
        double result = 0;
        cublasDnrm2(device::get_handle(), v.size(), v.data(), 1, &result);
        return result;
    }

    double dot(vector<double> const& u, vector<double> const& v)
    {
        assert(u.size() == v.size());

        double result = 0;
        cublasDdot(device::get_handle(), u.size(), u.data(), 1, v.data(), 1, &result);
        return result;
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

        double alpha = 1;
        double beta = 1;
        cublasDgemv(device::get_handle(), CUBLAS_OP_N,
            u.rows(), u.cols(), &alpha, u.data(), u.cols(),
            v.data(), 1, &beta, result.data(), 1);

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
