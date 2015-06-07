#ifndef LA_H
#define LA_H

#include <vector>

namespace la {

    void imul(std::vector<double>& u, double d);

    void iadd(std::vector<double>& u, std::vector<double> const& v);
    void isub(std::vector<double>& u, std::vector<double> const& v);
    void imul(std::vector<double>& u, std::vector<double> const& v);
    void idiv(std::vector<double>& u, std::vector<double> const& v);

    double dot(std::vector<double> const& u, std::vector<double> const& v);
}

#endif
