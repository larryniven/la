#include "la/la-cpu.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <cblas.h>

namespace la {

    namespace cpu {

        // vector operations

        void copy(vector_like<double>& u, vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            std::copy(v.data(), v.data() + v.size(), u.data());
        }

        void zero(vector_like<double>& u)
        {
            std::memset(u.data(), 0, u.size() * sizeof(double));
        }

        void axpy(vector_like<double>& y, double a, vector_like<double> const& x)
        {
            assert(y.size() == x.size());

            // cblas_daxpy(y.size(), a, x.data(), 1, y.data(), 1);

            unsigned int size = x.size();
            double *y_data = y.data();
            double const *x_data = x.data();

            for (int i = 0; i < size; ++i) {
                y_data[i] += a * x_data[i];
            }
        }

        void emul(vector_like<double>& z,
            vector_like<double> const& u,
            vector_like<double> const& v)
        {
            assert(u.size() == v.size() && z.size() == v.size());

            for (int i = 0; i < z.size(); ++i) {
                z(i) += u(i) * v(i);
            }
        }

        void div(vector_like<double>& z, double d, vector_like<double> const& v)
        {
            assert(z.size() == v.size());

            for (int i = 0; i < z.size(); ++i) {
                z(i) += d / v(i);
            }
        }

        void ediv(vector_like<double>& z,
            vector_like<double> const& u, vector_like<double> const& v)
        {
            assert(z.size() == u.size() && u.size() == v.size());

            for (int i = 0; i < v.size(); ++i) {
                z(i) += u(i) / v(i);
            }
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

        bool has_nan(vector_like<double> const& u)
        {
            for (int i = 0; i < u.size(); ++i) {
                if (std::isnan(u(i))) {
                    return true;
                }
            }

            return false;
        }

        void exp(vector_like<double>& z, vector_like<double> const& u)
        {
            assert(z.size() == u.size());

            for (int i = 0; i < z.size(); ++i) {
                z(i) += std::exp(u(i));
            }
        }

        void log(vector_like<double>& z, vector_like<double> const& u)
        {
            assert(z.size() == u.size());

            for (int i = 0; i < z.size(); ++i) {
                z(i) += std::log(u(i));
            }
        }

        // matrix operations

        void copy(matrix_like<double>& u, matrix_like<double> const& v)
        {
            assert(u.rows() == v.rows() && u.cols() == v.cols());

            std::copy(v.data(), v.data() + v.rows() * v.cols(), u.data());
        }

        void zero(matrix_like<double>& m)
        {
            std::memset(m.data(), 0, m.rows() * m.cols() * sizeof(double));
        }

        void axpy(matrix_like<double>& y, double a, matrix_like<double> const& x)
        {
            assert(y.rows() == x.rows() && y.cols() == x.cols());

            cblas_daxpy(y.rows() * y.cols(), a, x.data(), 1, y.data(), 1);
        }

        void emul(matrix_like<double>& z,
            matrix_like<double> const& u, matrix_like<double> const& v)
        {
            emul(z.as_vector(), u.as_vector(), v.as_vector());
        }

        void div(matrix_like<double>& z, double d, matrix_like<double> const& u)
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
            if (!(u.size() == a.rows() && a.cols() == v.size())) {
                throw std::logic_error(
                    "assertion u.size() == a.rows() && a.cols() == v.size() fails\n"
                    "u.size() = " + std::to_string(u.size()) + "\n"
                    "a.rows() = " + std::to_string(a.rows()) + "\n"
                    "a.cols() = " + std::to_string(a.cols()) + "\n"
                    "v.size() = " + std::to_string(v.size())
                );
            }

            cblas_dgemv(CblasRowMajor, CblasNoTrans, a.rows(), a.cols(), 1, a.data(), a.cols(),
                v.data(), 1, 1, u.data(), 1);
        }

        void lmul(vector_like<double>& u,
            vector_like<double> const& v,
            matrix_like<double> const& a)
        {
            if (!(u.size() == a.cols() && a.rows() == v.size())) {
                throw std::logic_error(
                    "assertion u.size() == a.rows() && a.cols() == v.size() fails\n"
                    "u.size() = " + std::to_string(u.size()) + "\n"
                    "a.rows() = " + std::to_string(a.rows()) + "\n"
                    "a.cols() = " + std::to_string(a.cols()) + "\n"
                    "v.size() = " + std::to_string(v.size())
                );
            }

            cblas_dgemv(CblasRowMajor, CblasTrans, a.rows(), a.cols(), 1, a.data(), a.cols(),
                v.data(), 1, 1, u.data(), 1);
        }

        void mul(matrix_like<double>& u, matrix_like<double> const& a,
            matrix_like<double> const& b)
        {
            assert(a.cols() == b.rows());
            assert(a.rows() == u.rows() && b.cols() == u.cols());

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, u.rows(), u.cols(), a.cols(),
                1, a.data(), a.cols(), b.data(), b.cols(), 1, u.data(), u.cols());
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

        void vdot(vector_like<double>& v, matrix_like<double> const& a,
            matrix_like<double> const& b)
        {
            assert(a.rows() == b.rows() && a.cols() == b.cols() && v.size() == a.rows());

            for (int i = 0; i < a.rows(); ++i) {
                weak_vector<double> a_vec {const_cast<double *>(a.data()) + i * a.cols(), a.cols()};
                weak_vector<double> b_vec {const_cast<double *>(b.data()) + i * b.cols(), b.cols()};

                v(i) = dot(a_vec, b_vec);
            }
        }

        double norm(matrix_like<double> const& m)
        {
            return norm(weak_vector<double> { const_cast<double*>(m.data()), m.rows() * m.cols() });
        }

        void outer_prod(matrix_like<double>& result,
            vector_like<double> const& a,
            vector_like<double> const& b)
        {
            assert(result.rows() == a.size() && result.cols() == b.size());

            cblas_dger(CblasRowMajor, a.size(), b.size(), 1, a.data(), 1, b.data(), 1,
                result.data(), b.size());
        }

        bool has_nan(matrix_like<double> const& m)
        {
            return has_nan(weak_vector<double> { const_cast<double*>(m.data()), m.rows() * m.cols() });
        }

        void exp(matrix_like<double>& z, matrix_like<double> const& u)
        {
            exp(z.as_vector(), u.as_vector());
        }

        void log(matrix_like<double>& z, matrix_like<double> const& u)
        {
            log(z.as_vector(), u.as_vector());
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

        void ediv(tensor_like<double>& z,
            tensor_like<double> const& u, tensor_like<double> const& v)
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

        double dot(tensor_like<double> const& a, tensor_like<double> const& b)
        {
            return dot(a.as_vector(), b.as_vector());
        }

        double norm(tensor_like<double> const& v)
        {
            return norm(v.as_vector());
        }

        void vdot(tensor_like<double>& v, tensor_like<double> const& a,
            tensor_like<double> const& b)
        {
            vdot(v.as_vector(), a.as_matrix(), b.as_matrix());
        }

        bool has_nan(tensor_like<double> const& a)
        {
            return has_nan(a.as_vector());
        }

        void exp(tensor_like<double>& z, tensor_like<double> const& u)
        {
            exp(z.as_vector(), u.as_vector());
        }

        void log(tensor_like<double>& z, tensor_like<double> const& u)
        {
            log(z.as_vector(), u.as_vector());
        }

        void corr_linearize(tensor_like<double>& result,
            tensor_like<double> const& u,
            int f1, int f2, int p1, int p2, int d1, int d2)
        {
            double *result_data = result.data();
            double const *u_data = u.data();

            assert(u.dim() == 4);

            unsigned int s0, s1, s2, s3;
            std::tie(s0, s1, s2, s3) = std::make_tuple(u.size(0), u.size(1),
                u.size(2), u.size(3));

            int result_vec_size = result.vec_size();
            int u_vec_size = u.vec_size();

            unsigned int r0 = s1 - f1 + 1 + 2 * p1;
            unsigned int r1 = s2 - f2 + 1 + 2 * p2;

            for (int n = 0; n < s0; ++n) {
                for (int i = 0; i < r0; ++i) {
                    for (int j = 0; j < r1; ++j) {
                        for (int a = 0; a < f1; ++a) {
                            for (int b = 0; b < f2; ++b) {

                                // int c1 = i + (a - (f1 / 2)) * d1;
                                // int c2 = j + (b - (f2 / 2)) * d2;

                                int c1 = i + (a - p1) * d1;
                                int c2 = j + (b - p2) * d2;

                                if (c1 < 0 || c2 < 0 || c1 >= s1 || c2 >= s2) {
                                    continue;
                                }

                                int output_base = n * r0 * r1 * f1 * f2 * s3 + i * r1 * f1 * f2 * s3
                                    + j * f1 * f2 * s3 + a * f2 * s3 + b * s3;
                                int input_base = n * s1 * s2 * s3 + c1 * s2 * s3 + c2 * s3;

                                for (int k = 0; k < s3; ++k) {
                                    assert(output_base + k < result_vec_size);
                                    assert(input_base + k < u_vec_size);

                                    result_data[output_base + k] += u_data[input_base + k];
                                }
                            }
                        }

                    }
                }
            }
        }

        void corr_linearize(tensor_like<double>& result,
            tensor_like<double> const& u,
            int f1, int f2)
        {
            corr_linearize(result, u, f1, f2, 0, 0, 1, 1);
        }

        void corr_delinearize(tensor_like<double>& result,
            tensor_like<double> const& u,
            int f1, int f2, int p1, int p2, int d1, int d2)
        {
            double *result_data = result.data();
            double const *u_data = u.data();

            assert(result.dim() == 4);

            unsigned int s0, s1, s2, s3;
            std::tie(s0, s1, s2, s3) = std::make_tuple(result.size(0), result.size(1),
                result.size(2), result.size(3));

            int u_vec_size = u.vec_size();
            int result_vec_size = result.vec_size();

            unsigned int r0 = s1 - f1 + 1 + 2 * p1;
            unsigned int r1 = s2 - f2 + 1 + 2 * p2;

            for (int n = 0; n < s0; ++n) {
                for (int i = 0; i < r0; ++i) {
                    for (int j = 0; j < r1; ++j) {
                        for (int a = 0; a < f1; ++a) {
                            for (int b = 0; b < f2; ++b) {

                                // int c1 = i + (a - (f1 / 2)) * d1;
                                // int c2 = j + (b - (f2 / 2)) * d2;

                                int c1 = i + (a - p1) * d1;
                                int c2 = j + (b - p2) * d2;

                                if (c1 < 0 || c2 < 0 || c1 >= s1 || c2 >= s2) {
                                    continue;
                                }

                                int input_base = n * r0 * r1 * f1 * f2 * s3 + i * r1 * f1 * f2 * s3
                                    + j * f1 * f2 * s3 + a * f2 * s3 + b * s3;
                                int output_base = n * s1 * s2 * s3 + c1 * s2 * s3 + c2 * s3;

                                for (int k = 0; k < s3; ++k) {
                                    assert(output_base + k < result_vec_size);
                                    assert(input_base + k < u_vec_size);

                                    result_data[output_base + k] += u_data[input_base + k];
                                }
                            }
                        }
                    }
                }
            }
        }

    }
}
