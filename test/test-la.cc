#include <vector>
#include <functional>
#include "la/la-cpu.h"
#include "ebt/ebt.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests = {
    {"test-size", []() {
        la::cpu::matrix<double> a {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        ebt::assert_equals(3, a.rows());
        ebt::assert_equals(3, a.cols());
        la::cpu::vector<double> b {1, 2, 3};
        ebt::assert_equals(3, b.size());
    }},

    {"test-mul", []() {
        la::cpu::matrix<double> a {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        la::cpu::vector<double> b {1, 2, 3};

        la::cpu::vector<double> r;
        r.resize(a.cols());

        mul(r, a, b);

        ebt::assert_equals(14, r(0));
        ebt::assert_equals(32, r(1));
        ebt::assert_equals(50, r(2));
    }},

    {"test-outer-product", []() {
        la::cpu::vector<double> a {1, 2};
        la::cpu::vector<double> b {3, 4, 5};

        la::cpu::matrix<double> c;
        c.resize(a.size(), b.size());

        outer_prod(c, a, b);

        ebt::assert_equals(3, c(0, 0));
        ebt::assert_equals(4, c(0, 1));
        ebt::assert_equals(5, c(0, 2));

        ebt::assert_equals(6, c(1, 0));
        ebt::assert_equals(8, c(1, 1));
        ebt::assert_equals(10, c(1, 2));
    }},

    {"test-tensor", []() {
        la::cpu::vector<double> v {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};

        la::cpu::weak_tensor<double> t {v.data(), std::vector<unsigned int>{2, 3, 4}};

        ebt::assert_equals(v(0), t({0, 0, 0}));
        ebt::assert_equals(v(1), t({0, 0, 1}));

        ebt::assert_equals(v(4), t({0, 1, 0}));
        ebt::assert_equals(v(5), t({0, 1, 1}));

        ebt::assert_equals(v(12), t({1, 0, 0}));
        ebt::assert_equals(v(13), t({1, 0, 1}));
    }},

    {"test-large-emul", []() {
        la::cpu::tensor<double> v;
        v.resize({474 * 41 * 61 * 61}, 1);

        la::cpu::tensor<double> u;
        u.resize({474 * 41 * 61 * 61}, 2);

        la::cpu::tensor<double> z;
        z.resize({474 * 41 * 61 * 61});

        la::cpu::emul(z, u, v);

        ebt::assert_equals(2, z({0}));
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
