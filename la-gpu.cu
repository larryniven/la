#include "la/la-gpu.h"
#include <cmath>
#include <cassert>
#include <cublas_v2.h>
#include <cblas.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

namespace la {

    namespace gpu {

        __global__ void print_vec(double const *p, int size)
        {
            for (int i = 0; i < size; ++i) {
                printf("%f ", p[i]);
            }
            printf("\n");
        }

        __global__ void print_mat(double const *p, int rows, int cols)
        {
            printf("%d %d\n", rows, cols);
        
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    printf("%f ", p[j * rows + i]);
                }
                printf("\n");
            }
        }

        device device::d = device();

        device::device()
        {
            cublasCreate(&handle);
        }

        device::~device()
        {
            cublasDestroy(handle);
            cudaDeviceReset();
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

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.begin()), thrust::device_ptr<double const>(v.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.end()), thrust::device_ptr<double const>(v.end()))),
                imul_op());
        }

        void idiv(vector<double>& u, vector<double> const& v)
        {
            assert(u.size() == v.size());

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.begin()), thrust::device_ptr<double const>(v.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.end()), thrust::device_ptr<double const>(v.end()))),
                idiv_op());
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

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(result.begin()), thrust::device_ptr<double const>(v.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(result.end()), thrust::device_ptr<double const>(v.end()))),
                ilogistic_op());

            return result;
        }

        void iadd(matrix<double>& u, matrix<double> const& v)
        {
            assert(u.rows() == v.rows());
            assert(u.cols() == v.cols());

            double alpha = 1;
            cublasDaxpy(device::get_handle(), u.rows() * u.cols(), &alpha, v.data(), 1, u.data(), 1);
        }

        void isub(matrix<double>& u, matrix<double> const& v)
        {
            assert(u.rows() == v.rows());
            assert(u.cols() == v.cols());

            double alpha = -1;
            cublasDaxpy(device::get_handle(), u.rows() * u.cols(), &alpha, v.data(), 1, u.data(), 1);
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
                u.rows(), u.cols(), &alpha, u.data(), u.rows(),
                v.data(), 1, &beta, result.data(), 1);

            return result;
        }

        vector<double> lmult(
            matrix<double> const& u,
            vector<double> const& v)
        {
            vector<double> result;
            result.resize(u.cols());
    
            double alpha = 1;
            double beta = 1;
            cublasDgemv(device::get_handle(), CUBLAS_OP_T,
                u.rows(), u.cols(), &alpha, u.data(), u.rows(),
                v.data(), 1, &beta, result.data(), 1);
    
            return result;
        }
 
        vector<double> tensor_prod(vector<double> const& a,
            vector<double> const& b)
        {
            vector<double> result;
            result.resize(a.size() * b.size());

            double alpha = 1;
            cublasDger(device::get_handle(), b.size(), a.size(),
                &alpha, b.data(), 1, a.data(), 1,
                result.data(), b.size());

            return result;
        }

        matrix<double> outer_prod(vector<double> const& a,
            vector<double> const& b)
        {
            matrix<double> result;
            result.resize(a.size(), b.size());

            double alpha = 1;
            cublasDger(device::get_handle(), a.size(), b.size(),
                &alpha, a.data(), 1, b.data(), 1,
                result.data(), a.size());

            return result;
        }
    }
}
