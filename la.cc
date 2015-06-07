#include "la/la.h"

namespace la {

    void imul(std::vector<double>& u, double d)
    {
        for (int i = 0; i < u.size(); ++i) {
            u[i] *= d;
        }
    }

    void iadd(std::vector<double>& u, std::vector<double> const& v)
    {
        for (int i = 0; i < v.size(); ++i) {
            u[i] += v[i];
        }
    }

    void isub(std::vector<double>& u, std::vector<double> const& v)
    {
        for (int i = 0; i < v.size(); ++i) {
            u[i] -= v[i];
        }
    }

    void imul(std::vector<double>& u, std::vector<double> const& v)
    {
        for (int i = 0; i < v.size(); ++i) {
            u[i] *= v[i];
        }
    }

    void idiv(std::vector<double>& u, std::vector<double> const& v)
    {
        for (int i = 0; i < v.size(); ++i) {
            u[i] /= v[i];
        }
    }

    double dot(std::vector<double> const& u, std::vector<double> const& v)
    {
        double sum = 0;
        for (int i = 0; i < u.size(); ++i) {
            sum += u[i] * v[i];
        }
        return sum;
    }

}
