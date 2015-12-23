#include <vector>
#include <functional>
#include "la/la.h"
#include "ebt/ebt.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests = {
    {"test-size", []() {
        la::matrix<double> a {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        ebt::assert_equals(3, a.rows());
        ebt::assert_equals(3, a.cols());
        la::vector<double> b {1, 2, 3};
        ebt::assert_equals(3, b.size());
    }},

    {"test-mult", []() {
        la::matrix<double> a {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        la::vector<double> b {1, 2, 3};
        la::vector<double> r = mult(a, b);
        ebt::assert_equals(3, r.size());
        ebt::assert_equals(14, r(0));
        ebt::assert_equals(32, r(1));
        ebt::assert_equals(50, r(2));
    }},

    {"test-tensor-product", []() {
        la::vector<double> a {1, 2};
        la::vector<double> b {3, 4, 5};
        la::vector<double> c = tensor_prod(a, b);

        ebt::assert_equals(6, c.size());

        ebt::assert_equals(3, c(0));
        ebt::assert_equals(4, c(1));
        ebt::assert_equals(5, c(2));

        ebt::assert_equals(6, c(3));
        ebt::assert_equals(8, c(4));
        ebt::assert_equals(10, c(5));
    }},

    {"test-outer-product", []() {
        la::vector<double> a {1, 2};
        la::vector<double> b {3, 4, 5};
        la::matrix<double> c = outer_prod(a, b);

        ebt::assert_equals(2, c.rows());
        ebt::assert_equals(3, c.cols());

        ebt::assert_equals(3, c(0, 0));
        ebt::assert_equals(4, c(0, 1));
        ebt::assert_equals(5, c(0, 2));

        ebt::assert_equals(6, c(1, 0));
        ebt::assert_equals(8, c(1, 1));
        ebt::assert_equals(10, c(1, 2));
    }},
};

int main()
{
    for (auto& t: tests) {
        std::cout << t.first << std::endl;
        t.second();
    }

    return 0;
}
