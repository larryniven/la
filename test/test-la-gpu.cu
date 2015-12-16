#include "ebt/ebt.h"
#include "la/la-gpu.h"
#include <vector>
#include <functional>

std::vector<std::function<void(void)>> tests = {
    []() {
        la::matrix<double> ha {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
        la::gpu::matrix<double> da {ha};
        ebt::assert_equals(3, da.rows());
        ebt::assert_equals(3, da.cols());
        la::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db {hb};
        ebt::assert_equals(3, hb.size());
    },

    /*
    []() {
        la::gpu::matrix<double> a {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
        la::gpu::vector<double> b {1, 2, 3};
        la::gpu::vector<double> r = mult(a, b);
        ebt::assert_equals(3, r.size());
        ebt::assert_equals(14, r(0));
        ebt::assert_equals(32, r(1));
        ebt::assert_equals(50, r(2));
    },
    */
};

int main()
{
    for (auto& t: tests) {
        t();
    }

    return 0;
}
