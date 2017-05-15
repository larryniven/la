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

        void imul(vector_like<double>& u, double d)
        {
            cublasDscal(device::get_handle(), u.size(), &d, u.data(), 1);
        }

        vector<double> mul(vector<double>&& u, double d)
        {
            vector<double> result { std::move(u) };
            imul(u, d);
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

            double alpha = -1;
            cublasDaxpy(device::get_handle(), u.size(), &alpha, v.data(), 1, u.data(), 1);
        }

        vector<double> sub(vector_like<double> const& u, vector_like<double> const& v)
        {
            vector<double> result { u };

            isub(result, v);

            return result;
        }

        struct idiv_op {
            template <class T>
            __host__ __device__
            void operator()(T t) const
            {
                thrust::get<0>(t) /= thrust::get<1>(t);
            }
        };

        void idiv(vector_like<double>& u, vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            thrust::for_each(thrust::device,
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.begin()), thrust::device_ptr<double const>(v.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(u.end()), thrust::device_ptr<double const>(v.end()))),
                idiv_op());
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

#if 0
            double alpha = 1;
            double beta = 1;
            cublasDgbmv(device::get_handle(), CUBLAS_OP_N, u.size(), u.size(), 0, 0,
                &alpha, u.data(), 1, v.data(), 1, &beta, z.data(), 1);
#endif
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

        void axpy(vector_like<double>& y, double a, vector_like<double> const& x)
        {
            assert(y.size() == x.size());

            cublasDaxpy(device::get_handle(), y.size(), &a, x.data(), 1, y.data(), 1);
        }

        // matrix operations

        void copy(matrix_like<double>& u, matrix_like<double> const& v)
        {
            assert(u.rows() == v.rows() && u.cols() == v.cols());

            cudaMemcpy(u.data(), v.data(), v.rows() * v.cols() * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        void zero(matrix_like<double>& m)
        {
            cudaMemset(m.data(), 0, m.rows() * m.cols() * sizeof(double));
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

            weak_vector<double> a {u.data(), u.rows() * u.cols()};
            weak_vector<double> b {const_cast<double*>(v.data()), v.rows() * v.cols()};

            iadd(a, b);
        }

        void isub(matrix_like<double>& u, matrix_like<double> const& v)
        {
            assert(u.rows() == v.rows());
            assert(u.cols() == v.cols());

            weak_vector<double> a {u.data(), u.rows() * u.cols()};
            weak_vector<double> b {const_cast<double*>(v.data()), v.rows() * v.cols()};

            isub(a, b);
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

        vector<double> mul(
            matrix_like<double> const& a,
            vector_like<double> const& v)
        {
            vector<double> result;
            result.resize(a.rows());

            mul(result, a, v);

            return result;
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

            double alpha = 1;
            double beta = 1;

            cublasDgemm(device::get_handle(), CUBLAS_OP_N, CUBLAS_OP_N, u.cols(), u.rows(), a.cols(),
                &alpha, b.data(), b.cols(), a.data(), a.cols(), &beta, u.data(), u.cols());
        }

        matrix<double> mul(matrix_like<double> const& a,
            matrix_like<double> const& b)
        {
            matrix<double> result;
            result.resize(a.rows(), b.cols());
            mul(result, a, b);

            return result;
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
            weak_vector<double> u(const_cast<double *>(v.data()), v.rows() * v.cols());
            return norm(u);
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

        matrix<double> outer_prod(vector_like<double> const& a,
            vector_like<double> const& b)
        {
            matrix<double> result;
            result.resize(a.size(), b.size());

            outer_prod(result, a, b);

            return result;
        }

        bool has_nan(matrix_like<double> const& m)
        {
            weak_vector<double> v { const_cast<double*>(m.data()), m.rows() * m.cols() };
            return has_nan(v);
        }

        void axpy(matrix_like<double>& y, double a, matrix_like<double> const& x)
        {
            weak_vector<double> y_v { y.data(), y.rows() * y.cols() };
            weak_vector<double> x_v { const_cast<double*>(x.data()), x.rows() * x.cols() };

            axpy(y_v, a, x_v);
        }

        void copy(tensor_like<double>& u, tensor_like<double> const& v)
        {
            copy(u.as_vector(), v.as_vector());
        }

        void zero(tensor_like<double>& u)
        {
            std::memset(u.data(), 0, u.vec_size() * sizeof(double));
        }

        void imul(tensor_like<double>& u, double a)
        {
            imul(u.as_vector(), a);
        }

        void mul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& v)
        {
            if (a.dim() == 1) {
                lmul(u.as_vector(), a.as_vector(), v.as_matrix());
            } else {
                mul(u.as_matrix(), a.as_matrix(), v.as_matrix());
            }
        }

        void ltmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b)
        {
            if (a.dim() == 1 && b.dim() == 1) {
                outer_prod(u.as_matrix(), a.as_vector(), b.as_vector());
            } else {
                ltmul(u.as_matrix(), a.as_matrix(), b.as_matrix());
            }
        }

        void rtmul(tensor_like<double>& u, tensor_like<double> const& a,
            tensor_like<double> const& b)
        {
            if (a.dim() == 1) {
                mul(u.as_vector(), b.as_matrix(), a.as_vector());
            } else {
                rtmul(u.as_matrix(), a.as_matrix(), b.as_matrix());
            }
        }

        tensor<double> mul(tensor_like<double> const& m,
            double a)
        {
            tensor<double> result { m };

            imul(result, a);

            return result;
        }

        tensor<double> mul(tensor_like<double> const& a,
            tensor_like<double> const& v)
        {
            tensor<double> result;

            std::vector<unsigned int> sizes = a.sizes();
            sizes.pop_back();
            sizes.push_back(v.size(v.dim() - 1));

            result.resize(sizes);

            mul(result, a, v);

            return result;
        }

        void resize_as(tensor<double>& a, tensor_like<double> const& b, double value)
        {
            a.resize(b.sizes(), value);
        }

        void emul(tensor_like<double>& z, tensor_like<double> const& u,
            tensor_like<double> const& v)
        {
            emul(z.as_vector(), u.as_vector(), v.as_vector());
        }

        void iadd(tensor_like<double>& a, tensor_like<double> const& b)
        {
            iadd(a.as_vector(), b.as_vector());
        }

        void isub(tensor_like<double>& a, tensor_like<double> const& b)
        {
            isub(a.as_vector(), b.as_vector());
        }

        double dot(tensor_like<double> const& a, tensor_like<double> const& b)
        {
            return dot(a.as_vector(), b.as_vector());
        }

        double norm(tensor_like<double> const& v)
        {
            return norm(v.as_vector());
        }

        bool has_nan(tensor_like<double> const& a)
        {
            return has_nan(a.as_vector());
        }

        void axpy(tensor_like<double>& y, double a, tensor_like<double> const& x)
        {
            axpy(y.as_vector(), a, x.as_vector());
        }

    }
}
