#include "la/la.h"
#include <cmath>
#include <cassert>

namespace la {

    void imul(std::vector<double>& u, double d)
    {
        for (int i = 0; i < u.size(); ++i) {
            u[i] *= d;
        }
    }

    void iadd(std::vector<double>& u, std::vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u[i] += v[i];
        }
    }

    void isub(std::vector<double>& u, std::vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u[i] -= v[i];
        }
    }

    void imul(std::vector<double>& u, std::vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u[i] *= v[i];
        }
    }

    void idiv(std::vector<double>& u, std::vector<double> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < v.size(); ++i) {
            u[i] /= v[i];
        }
    }

    double dot(std::vector<double> const& u, std::vector<double> const& v)
    {
        assert(u.size() == v.size());

        double sum = 0;
        for (int i = 0; i < u.size(); ++i) {
            sum += u[i] * v[i];
        }
        return sum;
    }

    void iadd(std::vector<std::vector<double>>& u, std::vector<std::vector<double>> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < u.size(); ++i) {
            iadd(u[i], v[i]);
        }
    }

    void isub(std::vector<std::vector<double>>& u, std::vector<std::vector<double>> const& v)
    {
        assert(u.size() == v.size());

        for (int i = 0; i < u.size(); ++i) {
            isub(u[i], v[i]);
        }
    }

    double norm(std::vector<double> const& v)
    {
        double sum = 0;

        for (int i = 0; i < v.size(); ++i) {
            sum += v[i] * v[i];
        }

        return std::sqrt(sum);
    }

    std::vector<double> logistic(std::vector<double> const& v)
    {
        std::vector<double> result;
        result.resize(v.size());

        for (int i = 0; i < v.size(); ++i) {
            result[i] = 1 / (1 + std::exp(-v[i]));
        }

        return result;
    }

    std::vector<double> add(
        std::vector<double> const& u,
        std::vector<double> const& v)
    {
        assert(u.size() == v.size());

        std::vector<double> result;
        result.resize(u.size());

        for (int i = 0; i < u.size(); ++i) {
            result[i] = u[i] + v[i];
        }

        return result;
    }

    std::vector<double> mult(
        std::vector<std::vector<double>> const& u,
        std::vector<double> const& v)
    {
        std::vector<double> result;
        result.resize(u.size());

        for (int i = 0; i < u.size(); ++i) {
            assert(u[i].size() == v.size());

            for (int j = 0; j < v.size(); ++j) {
                result[i] += u[i][j] * v[j];
            }
        }

        return result;
    }

}
