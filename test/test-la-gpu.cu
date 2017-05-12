#include "ebt/ebt.h"
#include "la/la-gpu.h"
#include <vector>
#include <functional>

std::vector<std::pair<std::string, std::function<void(void)>>> tests = {
    {"test-vec-copy", []() {
        la::cpu::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db {hb};
        ebt::assert_equals(3, db.size());

        la::cpu::vector<double> hb2 = to_host(db);
        ebt::assert_equals(1, hb2(0));
        ebt::assert_equals(2, hb2(1));
        ebt::assert_equals(3, hb2(2));
    }},

    {"test-vec-resize", []() {
        la::gpu::vector<double> db;
        db.resize(3, 1);
        la::cpu::vector<double> hb = to_host(db);
        ebt::assert_equals(1, hb(0));
        ebt::assert_equals(1, hb(1));
        ebt::assert_equals(1, hb(2));
    }},

    {"test-vec-has-nan", []() {
        la::cpu::vector<double> hv;
        hv.resize(3);
        hv(0) = 1;
        hv(1) = 2;
        hv(2) = std::numeric_limits<double>::quiet_NaN();
        la::gpu::vector<double> dv { hv };

        ebt::assert_equals(true, la::gpu::has_nan(dv));

        la::cpu::vector<double> hv2;
        hv2.resize(3);
        hv2(0) = 4;
        hv2(1) = 5;
        hv2(2) = 6;
        la::gpu::vector<double> dv2 { hv2 };

        ebt::assert_equals(false, la::gpu::has_nan(dv2));
    }},

    {"test-weak-vec", []() {
        la::cpu::vector<double> hb {1, 2, 3, 4, 5, 6};
        la::gpu::vector<double> db {hb};

        la::gpu::weak_vector<double> db2 { db.data() + 3, 3 };
        la::cpu::vector<double> hb2 = to_host(db2);
        ebt::assert_equals(4, hb2(0));
        ebt::assert_equals(5, hb2(1));
        ebt::assert_equals(6, hb2(2));
    }},

    {"test-vec-emul", []() {
        la::cpu::vector<double> ha {1, 2, 3};
        la::gpu::vector<double> da {ha};
        la::cpu::vector<double> hb {4, 5, 6};
        la::gpu::vector<double> db {hb};

        la::gpu::vector<double> dc = emul(da, db);
        la::cpu::vector<double> hc = to_host(dc);
        ebt::assert_equals(4, hc(0));
        ebt::assert_equals(10, hc(1));
        ebt::assert_equals(18, hc(2));
    }},

    {"test-mat-copy", []() {
        la::cpu::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};

        ebt::assert_equals(2, da.rows());
        ebt::assert_equals(3, da.cols());

        la::cpu::matrix<double> ha2 = to_host(da);

        ebt::assert_equals(1, ha2(0, 0));
        ebt::assert_equals(2, ha2(0, 1));
        ebt::assert_equals(3, ha2(0, 2));

        ebt::assert_equals(4, ha2(1, 0));
        ebt::assert_equals(5, ha2(1, 1));
        ebt::assert_equals(6, ha2(1, 2));
    }},

    {"test-weak-mat", []() {
        la::cpu::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};
        la::gpu::weak_matrix<double> da2 {da.data(), 2, 2};

        la::cpu::matrix<double> ha2 = to_host(da2);

        ebt::assert_equals(1, ha2(0, 0));
        ebt::assert_equals(2, ha2(0, 1));
        ebt::assert_equals(3, ha2(1, 0));
        ebt::assert_equals(4, ha2(1, 1));
    }},

    {"test-mat-iadd", []() {
        la::cpu::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};

        la::cpu::matrix<double> hb {{7, 8, 9}, {10, 11, 12}};
        la::gpu::matrix<double> db {hb};

        iadd(da, db);

        la::cpu::matrix<double> ha2 = to_host(da);

        ebt::assert_equals(8, ha2(0, 0));
        ebt::assert_equals(10, ha2(0, 1));
        ebt::assert_equals(12, ha2(0, 2));

        ebt::assert_equals(14, ha2(1, 0));
        ebt::assert_equals(16, ha2(1, 1));
        ebt::assert_equals(18, ha2(1, 2));
    }},

    {"test-mul", []() {
        la::cpu::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};
        la::cpu::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db {hb};
        la::gpu::vector<double> dr = mul(da, db);
        la::cpu::vector<double> hr = to_host(dr);
        ebt::assert_equals(2, hr.size());
        ebt::assert_equals(14, hr(0));
        ebt::assert_equals(32, hr(1));
    }},

    {"test-lmul", []() {
        la::cpu::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};
        la::cpu::vector<double> hb {1, 2};
        la::gpu::vector<double> db {hb};
        la::gpu::vector<double> dr = lmul(db, da);
        la::cpu::vector<double> hr = to_host(dr);
        ebt::assert_equals(3, hr.size());
        ebt::assert_equals(9, hr(0));
        ebt::assert_equals(12, hr(1));
        ebt::assert_equals(15, hr(2));
    }},

    {"test-matrix-mul", []() {
        la::cpu::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};
        la::cpu::matrix<double> hb {{7, 8, 9, 10}, {11, 12, 13, 14}, {15, 16, 17, 18}};
        la::gpu::matrix<double> db {hb};
        la::gpu::matrix<double> dr = mul(da, db);
        la::cpu::matrix<double> hr = to_host(dr);

        ebt::assert_equals(2, hr.rows());
        ebt::assert_equals(4, hr.cols());

        ebt::assert_equals(74, hr(0, 0));
        ebt::assert_equals(80, hr(0, 1));
        ebt::assert_equals(86, hr(0, 2));
        ebt::assert_equals(92, hr(0, 3));
        ebt::assert_equals(173, hr(1, 0));
        ebt::assert_equals(188, hr(1, 1));
        ebt::assert_equals(203, hr(1, 2));
        ebt::assert_equals(218, hr(1, 3));
    }},

    {"test-ltmul", []() {
        la::cpu::matrix<double> ha {{1, 4}, {2, 5}, {3, 6}};
        la::gpu::matrix<double> da {ha};
        la::cpu::matrix<double> hb {{7, 8, 9, 10}, {11, 12, 13, 14}, {15, 16, 17, 18}};
        la::gpu::matrix<double> db {hb};
        la::gpu::matrix<double> dr;
        dr.resize(2, 4);
        ltmul(dr, da, db);
        la::cpu::matrix<double> hr = to_host(dr);

        ebt::assert_equals(2, hr.rows());
        ebt::assert_equals(4, hr.cols());

        ebt::assert_equals(74, hr(0, 0));
        ebt::assert_equals(80, hr(0, 1));
        ebt::assert_equals(86, hr(0, 2));
        ebt::assert_equals(92, hr(0, 3));
        ebt::assert_equals(173, hr(1, 0));
        ebt::assert_equals(188, hr(1, 1));
        ebt::assert_equals(203, hr(1, 2));
        ebt::assert_equals(218, hr(1, 3));
    }},

    {"test-rtmul", []() {
        la::cpu::matrix<double> ha {{1, 2, 3}, {4, 5, 6}};
        la::gpu::matrix<double> da {ha};
        la::cpu::matrix<double> hb {{7, 11, 15}, {8, 12, 16}, {9, 13, 17}, {10, 14, 18}};
        la::gpu::matrix<double> db {hb};
        la::gpu::matrix<double> dr;
        dr.resize(2, 4);
        rtmul(dr, da, db);
        la::cpu::matrix<double> hr = to_host(dr);

        ebt::assert_equals(2, hr.rows());
        ebt::assert_equals(4, hr.cols());

        ebt::assert_equals(74, hr(0, 0));
        ebt::assert_equals(80, hr(0, 1));
        ebt::assert_equals(86, hr(0, 2));
        ebt::assert_equals(92, hr(0, 3));
        ebt::assert_equals(173, hr(1, 0));
        ebt::assert_equals(188, hr(1, 1));
        ebt::assert_equals(203, hr(1, 2));
        ebt::assert_equals(218, hr(1, 3));
    }},

    {"test-vec-tensor-prod", []() {
        la::cpu::vector<double> ha {1, 2};
        la::gpu::vector<double> da {ha};

        la::cpu::vector<double> hb {3, 4, 5};
        la::gpu::vector<double> db {hb};

        la::gpu::vector<double> dc = tensor_prod(da, db);
        la::cpu::vector<double> hc = to_host(dc);

        ebt::assert_equals(6, hc.size());

        ebt::assert_equals(3, hc(0));
        ebt::assert_equals(4, hc(1));
        ebt::assert_equals(5, hc(2));

        ebt::assert_equals(6, hc(3));
        ebt::assert_equals(8, hc(4));
        ebt::assert_equals(10, hc(5));
    }},

    {"test-outer-prod", []() {
        la::cpu::vector<double> ha {1, 2};
        la::gpu::vector<double> da {ha};

        la::cpu::vector<double> hb {3, 4, 5};
        la::gpu::vector<double> db {hb};

        la::gpu::matrix<double> dc = outer_prod(da, db);
        la::cpu::matrix<double> hc = to_host(dc);

        ebt::assert_equals(2, hc.rows());
        ebt::assert_equals(3, hc.cols());

        ebt::assert_equals(3, hc(0, 0));
        ebt::assert_equals(4, hc(0, 1));
        ebt::assert_equals(5, hc(0, 2));

        ebt::assert_equals(6, hc(1, 0));
        ebt::assert_equals(8, hc(1, 1));
        ebt::assert_equals(10, hc(1, 2));
    }},

    {"test-tensor", []() {
        la::cpu::vector<double> v {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};

        la::cpu::tensor<double> ht {v, std::vector<unsigned int>{2, 3, 4}};

        la::gpu::tensor<double> dt { ht };
        la::cpu::tensor<double> ht2 = to_host(dt);

        ebt::assert_equals(v(0), ht2({0, 0, 0}));
        ebt::assert_equals(v(1), ht2({0, 0, 1}));

        ebt::assert_equals(v(4), ht2({0, 1, 0}));
        ebt::assert_equals(v(5), ht2({0, 1, 1}));

        ebt::assert_equals(v(12), ht2({1, 0, 0}));
        ebt::assert_equals(v(13), ht2({1, 0, 1}));
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
