#include "la/la-gpu.h"
#include <cmath>
#include <cassert>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/logical.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

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

        void copy(vector_like<double>& u, vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            cudaMemcpy(u.data(), v.data(), v.size() * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        void zero(vector_like<double>& v)
        {
            cudaMemset(v.data(), 0, v.size() * sizeof(double));
        }

        void axpy(vector_like<double>& y, double a, vector_like<double> const& x)
        {
            assert(y.size() == x.size());

            cublasDaxpy(device::get_handle(), y.size(), &a, x.data(), 1, y.data(), 1);
        }

        struct emul_op {
            template <class T>
            __host__ __device__
            void operator()(T t) const
            {
                thrust::get<0>(t) += thrust::get<1>(t) * thrust::get<2>(t);
            }
        };

        void emul(vector_like<double>& z, vector_like<double> const& u,
            vector_like<double> const& v)
        {
            assert(u.size() == v.size() && z.size() == v.size());

            thrust::for_each(thrust::device,
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(z.begin()),
                    thrust::device_ptr<double const>(u.begin()),
                    thrust::device_ptr<double const>(v.begin())
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(z.end()),
                    thrust::device_ptr<double const>(u.end()),
                    thrust::device_ptr<double const>(v.end())
                )),
                emul_op());
        }

        struct div_op {
            double d;

            template <class T>
            __host__ __device__
            void operator()(T t) const
            {
                thrust::get<0>(t) += d / thrust::get<1>(t);
            }
        };

        void div(vector_like<double>& z, double d, vector_like<double> const& u)
        {
            assert(z.size() == u.size());

            thrust::for_each(thrust::device,
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(z.begin()),
                    thrust::device_ptr<double const>(u.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(z.end()),
                    thrust::device_ptr<double const>(u.end()))),
                div_op(d));
        }

        struct ediv_op {
            template <class T>
            __host__ __device__
            void operator()(T t) const
            {
                thrust::get<0>(t) += thrust::get<1>(t) / thrust::get<2>(t);
            }
        };

        void ediv(vector_like<double>& z,
            vector_like<double> const& u, vector_like<double> const& v)
        {
            assert(z.size() == u.size() && u.size() == v.size());

            thrust::for_each(thrust::device,
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(z.begin()),
                    thrust::device_ptr<double const>(u.begin()),
                    thrust::device_ptr<double const>(v.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(z.end()),
                    thrust::device_ptr<double const>(u.end()),
                    thrust::device_ptr<double const>(v.end()))),
                ediv_op());
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

        struct isnan_op {

            __host__ __device__
            bool operator()(double d)
            {
                return isnan(d);
            }
        };

        bool has_nan(vector_like<double> const& u)
        {
            thrust::device_ptr<double> p(const_cast<double*>(u.data()));

            return thrust::any_of(thrust::device, p, p + u.size(), isnan_op());
        }

        // matrix operations

        void copy(matrix_like<double>& u, matrix_like<double> const& v)
        {
            assert(u.rows() == v.rows() && u.cols() == v.cols());

            cudaMemcpy(u.data(), v.data(), v.rows() * v.cols() * sizeof(double),
                cudaMemcpyDeviceToDevice);
        }

        void zero(matrix_like<double>& m)
        {
            cudaMemset(m.data(), 0, m.rows() * m.cols() * sizeof(double));
        }

        void axpy(matrix_like<double>& y, double a, matrix_like<double> const& x)
        {
            axpy(y.as_vector(), a, x.as_vector());
        }

        void emul(matrix_like<double>& z, matrix_like<double> const& u,
            matrix_like<double> const& v)
        {
            emul(z.as_vector(), u.as_vector(), v.as_vector());
        }

        void div(matrix_like<double>& z, double d, matrix_like<double>& u)
        {
            div(z.as_vector(), d, u.as_vector());
        }

        void ediv(matrix_like<double>& z,
            matrix_like<double> const& u, matrix_like<double> const& v)
        {
            ediv(z.as_vector(), u.as_vector(), v.as_vector());
        }

        void mul(vector_like<double>& u, matrix_like<double> const& a,
            vector_like<double> const& v)
        {
            assert(u.size() == a.rows() && a.cols() == v.size());

            double alpha = 1;
            double beta = 1;
            cublasDgemv(device::get_handle(), CUBLAS_OP_T,
                a.cols(), a.rows(), &alpha, a.data(), a.cols(),
                v.data(), 1, &beta, u.data(), 1);
        }

        void lmul(vector_like<double>& u, 
            vector_like<double> const& v, matrix_like<double> const& a)
        {
            assert(u.size() == a.cols());

            double alpha = 1;
            double beta = 1;
            cublasDgemv(device::get_handle(), CUBLAS_OP_N,
                a.cols(), a.rows(), &alpha, a.data(), a.cols(),
                v.data(), 1, &beta, u.data(), 1);
        }

        void mul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b)
        {
            assert(a.cols() == b.rows());
            assert(a.rows() == u.rows() && b.cols() == u.cols());

            double alpha = 1;
            double beta = 1;

            cublasDgemm(device::get_handle(), CUBLAS_OP_N, CUBLAS_OP_N, u.cols(), u.rows(), a.cols(),
                &alpha, b.data(), b.cols(), a.data(), a.cols(), &beta, u.data(), u.cols());
        }

        void ltmul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b)
        {
            assert(a.rows() == b.rows());
            assert(u.rows() == a.cols() && u.cols() == b.cols());

            double alpha = 1;
            double beta = 1;

            cublasDgemm(device::get_handle(), CUBLAS_OP_N, CUBLAS_OP_T, u.cols(), u.rows(), a.rows(),
                &alpha, b.data(), b.cols(), a.data(), a.cols(), &beta, u.data(), u.cols());
        }

        void rtmul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b)
        {
            assert(a.cols() == b.cols());
            assert(u.rows() == a.rows() && u.cols() == b.rows());

            double alpha = 1;
            double beta = 1;

            cublasDgemm(device::get_handle(), CUBLAS_OP_T, CUBLAS_OP_N, u.cols(), u.rows(), a.cols(),
                &alpha, b.data(), b.cols(), a.data(), a.cols(), &beta, u.data(), u.cols());
        }

        double norm(matrix_like<double> const& v)
        {
            return norm(v.as_vector());
        }

        double dot(matrix_like<double> const& u, matrix_like<double> const& v)
        {
            return dot(u.as_vector(), v.as_vector());
        }

        void outer_prod(matrix_like<double>& result,
            vector_like<double> const& a,
            vector_like<double> const& b)
        {
            assert(result.rows() == a.size() && result.cols() == b.size());

            double alpha = 1;
            cublasDger(device::get_handle(), b.size(), a.size(),
                &alpha, b.data(), 1, a.data(), 1,
                result.data(), b.size());
        }

        bool has_nan(matrix_like<double> const& m)
        {
            return has_nan(m.as_vector());
        }

        // tensor operations

        void copy(tensor_like<double>& u, tensor_like<double> const& v)
        {
            copy(u.as_vector(), v.as_vector());
        }

        void zero(tensor_like<double>& u)
        {
            std::memset(u.data(), 0, u.vec_size() * sizeof(double));
        }

        void axpy(tensor_like<double>& y, double a, tensor_like<double> const& x)
        {
            axpy(y.as_vector(), a, x.as_vector());
        }

        void emul(tensor_like<double>& z, tensor_like<double> const& u,
            tensor_like<double> const& v)
        {
            emul(z.as_vector(), u.as_vector(), v.as_vector());
        }

        void div(tensor_like<double>& z, double d, tensor_like<double> const& u)
        {
            div(z.as_vector(), d, u.as_vector());
        }

        void ediv(tensor_like<double>& z, tensor_like<double> const& u,
            tensor_like<double> const& v)
        {
            ediv(z.as_vector(), u.as_vector(), v.as_vector());
        }

        void mul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& v)
        {
            matrix_like<double> const& a_mat = a.as_matrix();
            matrix_like<double> const& v_mat = v.as_matrix();

            if (a_mat.cols() == 1 && v_mat.rows() == 1) {
                outer_prod(u.as_matrix(), a.as_vector(), v.as_vector());
            } else if (a_mat.rows() == 1 && v_mat.cols() != 1) {
                lmul(u.as_vector(), a.as_vector(), v.as_matrix());
            } else if (a_mat.rows() != 1 && v_mat.cols() == 1) {
                mul(u.as_vector(), a.as_matrix(), v.as_vector());
            } else {
                mul(u.as_matrix(), a.as_matrix(), v.as_matrix());
            }
        }

        void ltmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b)
        {
            matrix_like<double> const& a_mat = a.as_matrix();
            matrix_like<double> const& b_mat = b.as_matrix();

            if (a_mat.rows() == 1 && b_mat.rows() == 1) {
                outer_prod(u.as_matrix(), a.as_vector(), b.as_vector());
            } else if (a_mat.rows() != 1 && b_mat.cols() == 1) {
                lmul(u.as_vector(), b.as_vector(), a.as_matrix());
            } else if (a_mat.cols() == 1 && b_mat.cols() != 1) {
                lmul(u.as_vector(), a.as_vector(), b.as_matrix());
            } else {
                ltmul(u.as_matrix(), a.as_matrix(), b.as_matrix());
            }
        }

        void rtmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b)
        {
            matrix_like<double> const& a_mat = a.as_matrix();
            matrix_like<double> const& b_mat = b.as_matrix();

            if (a_mat.cols() == 1 && b_mat.cols() == 1) {
                outer_prod(u.as_matrix(), a.as_vector(), b.as_vector());
            } else if (a_mat.rows() == 1 && b_mat.cols() != 1) {
                mul(u.as_vector(), b.as_matrix(), a.as_vector());
            } else if (a_mat.rows() != 1 && b_mat.rows() == 1) {
                mul(u.as_vector(), a.as_matrix(), b.as_vector());
            } else {
                rtmul(u.as_matrix(), a.as_matrix(), b.as_matrix());
            }
        }

        void resize_as(tensor<double>& a, tensor_like<double> const& b, double value)
        {
            a.resize(b.sizes(), value);
        }

        double norm(tensor_like<double> const& v)
        {
            return norm(v.as_vector());
        }

        double dot(tensor_like<double> const& a, tensor_like<double> const& b)
        {
            return dot(a.as_vector(), b.as_vector());
        }

        bool has_nan(tensor_like<double> const& a)
        {
            return has_nan(a.as_vector());
        }

    }
}
