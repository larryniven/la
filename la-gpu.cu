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

        // vector operations

        void zero(vector_like<double>& v)
        {
            cudaMemset(v.data(), 0, v.size() * sizeof(double));
        }

        void imul(vector_like<double>& u, double d)
        {
            cublasDscal(device::get_handle(), u.size(), &d, u.data(), 1);
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

            double alpha = 1;
            cublasDaxpy(device::get_handle(), u.size(), &alpha, v.data(), 1, u.data(), 1);
        }

        vector<double> add(
            vector_like<double> const& u,
            vector_like<double> const& v)
        {
            vector<double> result { u };
            iadd(result, v);
            return result;
        }

        void isub(vector_like<double>& u, vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            double alpha = -1;
            cublasDaxpy(device::get_handle(), u.size(), &alpha, v.data(), 1, u.data(), 1);
        }

        void idiv(vector_like<double>& u, vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.begin()), thrust::device_ptr<double const>(v.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.end()), thrust::device_ptr<double const>(v.end()))),
                idiv_op());
        }

        void emul(vector_like<double>& z, vector_like<double> const& u,
            vector_like<double> const& v)
        {
            assert(u.size() == v.size() && z.size() == v.size());

            double alpha = 1;
            double beta = 1;
            cublasDgbmv(device::get_handle(), CUBLAS_OP_N, u.size(), u.size(), 0, 0,
                &alpha, u.data(), 1, v.data(), 1, &beta, z.data(), 1);
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
            double result = 0;
            cublasDnrm2(device::get_handle(), v.size(), v.data(), 1, &result);
            return result;
        }

        double dot(vector_like<double> const& u, vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            double result = 0;
            cublasDdot(device::get_handle(), u.size(), u.data(), 1, v.data(), 1, &result);
            return result;
        }

        // matrix operations

        void zero(matrix_like<double>& m)
        {
            cudaMemset(m.data(), 0, m.rows() * m.cols() * sizeof(double));
        }

        void iadd(matrix_like<double>& u, matrix_like<double> const& v)
        {
            assert(u.rows() == v.rows());
            assert(u.cols() == v.cols());

            double alpha = 1;
            cublasDaxpy(device::get_handle(), u.rows() * u.cols(), &alpha, v.data(), 1, u.data(), 1);
        }

        void isub(matrix_like<double>& u, matrix_like<double> const& v)
        {
            assert(u.rows() == v.rows());
            assert(u.cols() == v.cols());

            double alpha = -1;
            cublasDaxpy(device::get_handle(), u.rows() * u.cols(), &alpha, v.data(), 1, u.data(), 1);
        }

        void mul(vector_like<double>& u, matrix_like<double> const& a,
            vector_like<double> const& v)
        {
            assert(u.size() == a.rows() && a.cols() == v.size());

            double alpha = 1;
            double beta = 1;
            cublasDgemv(device::get_handle(), CUBLAS_OP_N,
                a.rows(), a.cols(), &alpha, a.data(), a.rows(),
                v.data(), 1, &beta, u.data(), 1);
        }

        vector<double> mul(
            matrix_like<double> const& a,
            vector_like<double> const& v)
        {
            vector<double> result;
            result.resize(a.rows());

            mul(result, a, v);

            return result;
        }

        vector<double> lmul(
            matrix_like<double> const& u,
            vector_like<double> const& v)
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
 
        vector<double> tensor_prod(vector_like<double> const& a,
            vector_like<double> const& b)
        {
            vector<double> result;
            result.resize(a.size() * b.size());

            double alpha = 1;
            cublasDger(device::get_handle(), b.size(), a.size(),
                &alpha, b.data(), 1, a.data(), 1,
                result.data(), b.size());

            return result;
        }

        matrix<double> outer_prod(vector_like<double> const& a,
            vector_like<double> const& b)
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
