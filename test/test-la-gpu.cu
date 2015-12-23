#include "ebt/ebt.h"
#include "la/la-gpu.h"
#include <vector>
#include <functional>

std::vector<std::pair<std::string, std::function<void(void)>>> tests = {
    {"test-vec-copy", []() {
        la::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db {hb};
        ebt::assert_equals(3, db.size());

        la::vector<double> hb2 = to_host(db);
        ebt::assert_equals(1, hb2(0));
        ebt::assert_equals(2, hb2(1));
        ebt::assert_equals(3, hb2(2));
    }},

    {"test-vec-resize", []() {
        la::gpu::vector<double> db;
        db.resize(3, 1);
        la::vector<double> hb = to_host(db);
        ebt::assert_equals(1, hb(0));
        ebt::assert_equals(1, hb(1));
        ebt::assert_equals(1, hb(2));
    }},

    {"test-vec-imul", []() {
        la::vector<double> ha {1, 2, 3};
        la::gpu::vector<double> da {ha};
        la::vector<double> hb {4, 5, 6};
        la::gpu::vector<double> db {hb};
        imul(da, db);
        la::vector<double> ha2 = to_host(da);
        ebt::assert_equals(4, ha2(0));
        ebt::assert_equals(10, ha2(1));
        ebt::assert_equals(18, ha2(2));
    }},

    {"test-mat-copy", []() {
        la::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};

        ebt::assert_equals(2, da.rows());
        ebt::assert_equals(3, da.cols());

        la::matrix<double> ha2 = to_host(da);

        ebt::assert_equals(1, ha2(0, 0));
        ebt::assert_equals(2, ha2(0, 1));
        ebt::assert_equals(3, ha2(0, 2));

        ebt::assert_equals(4, ha2(1, 0));
        ebt::assert_equals(5, ha2(1, 1));
        ebt::assert_equals(6, ha2(1, 2));
    }},

    {"test-mult", []() {
        la::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};
        la::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db {hb};
        la::gpu::vector<double> dr = mult(da, db);
        la::vector<double> hr = to_host(dr);
        ebt::assert_equals(2, hr.size());
        ebt::assert_equals(14, hr(0));
        ebt::assert_equals(32, hr(1));
    }},

    {"test-lmult", []() {
        la::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};
        la::vector<double> hb {1, 2};
        la::gpu::vector<double> db {hb};
        la::gpu::vector<double> dr = lmult(da, db);
        la::vector<double> hr = to_host(dr);
        ebt::assert_equals(3, hr.size());
        ebt::assert_equals(9, hr(0));
        ebt::assert_equals(12, hr(1));
        ebt::assert_equals(15, hr(2));
    }},

    {"test-mat-iadd", []() {
        la::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};

        la::matrix<double> hb {{7, 8, 9}, {10, 11, 12}};
        la::gpu::matrix<double> db {hb};

        iadd(da, db);

        la::matrix<double> ha2 = to_host(da);

        ebt::assert_equals(8, ha2(0, 0));
        ebt::assert_equals(10, ha2(0, 1));
        ebt::assert_equals(12, ha2(0, 2));

        ebt::assert_equals(14, ha2(1, 0));
        ebt::assert_equals(16, ha2(1, 1));
        ebt::assert_equals(18, ha2(1, 2));
    }},

    {"test-vec-tensor-prod", []() {
        la::vector<double> ha {1, 2};
        la::gpu::vector<double> da {ha};

        la::vector<double> hb {3, 4, 5};
        la::gpu::vector<double> db {hb};

        la::gpu::vector<double> dc = tensor_prod(da, db);
        la::vector<double> hc = to_host(dc);

        ebt::assert_equals(6, hc.size());

        ebt::assert_equals(3, hc(0));
        ebt::assert_equals(4, hc(1));
        ebt::assert_equals(5, hc(2));

        ebt::assert_equals(6, hc(3));
        ebt::assert_equals(8, hc(4));
        ebt::assert_equals(10, hc(5));
    }},

    {"test-outer-prod", []() {
        la::vector<double> ha {1, 2};
        la::gpu::vector<double> da {ha};

        la::vector<double> hb {3, 4, 5};
        la::gpu::vector<double> db {hb};

        la::gpu::matrix<double> dc = outer_prod(da, db);
        la::matrix<double> hc = to_host(dc);

        ebt::assert_equals(2, hc.rows());
        ebt::assert_equals(3, hc.cols());

        ebt::assert_equals(3, hc(0, 0));
        ebt::assert_equals(4, hc(0, 1));
        ebt::assert_equals(5, hc(0, 2));

        ebt::assert_equals(6, hc(1, 0));
        ebt::assert_equals(8, hc(1, 1));
        ebt::assert_equals(10, hc(1, 2));
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
