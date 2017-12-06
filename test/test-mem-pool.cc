#include <vector>
#include <functional>
#include "la/mem-pool.h"
#include "ebt/assert.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests = {
    {"test-init", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        ebt::assert_equals(15, pool.used.size());
        ebt::assert_equals(8, pool.block.size());
    }},

    {"test-allocate-all", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p = pool.malloc(8 * (1 << 10));

        ebt::assert_equals((void *) 0, p);
        ebt::assert_equals(8, pool.used[0]);
        ebt::assert_equals(0, pool.block[0]);
    }},

    {"test-alloc-half", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p = pool.malloc(4 * (1 << 10));

        ebt::assert_equals((void *) 0, p);
        ebt::assert_equals(4, pool.used[0]);
        ebt::assert_equals(4, pool.used[1]);
        ebt::assert_equals(1, pool.block[0]);
    }},

    {"test-alloc-one", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p = pool.malloc(1 << 10);

        ebt::assert_equals((void *) 0, p);
        ebt::assert_equals(1, pool.used[0]);
        ebt::assert_equals(1, pool.used[1]);
        ebt::assert_equals(1, pool.used[3]);
        ebt::assert_equals(1, pool.used[7]);
        ebt::assert_equals(7, pool.block[0]);
    }},

    {"test-alloc-smaller-than-one", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p = pool.malloc(1 << 9);

        ebt::assert_equals((void *) 0, p);
        ebt::assert_equals(1, pool.used[0]);
        ebt::assert_equals(1, pool.used[1]);
        ebt::assert_equals(1, pool.used[3]);
        ebt::assert_equals(1, pool.used[7]);
        ebt::assert_equals(7, pool.block[0]);
    }},

    {"test-alloc-two", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p1 = pool.malloc(1 << 10);
        void *p2 = pool.malloc(1 << 10);

        ebt::assert_equals(2, pool.used[0]);
        ebt::assert_equals(2, pool.used[1]);
        ebt::assert_equals(2, pool.used[3]);
        ebt::assert_equals(1, pool.used[7]);
        ebt::assert_equals(1, pool.used[8]);
        ebt::assert_equals(7, pool.block[0]);
        ebt::assert_equals(8, pool.block[1]);
        ebt::assert_equals((void *) 0, p1);
        ebt::assert_equals((void *) (1 << 10), p2);
    }},

    {"test-alloc-one-two", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p1 = pool.malloc(1 << 10);
        void *p2 = pool.malloc(2 * (1 << 10));

        ebt::assert_equals(3, pool.used[0]);
        ebt::assert_equals(3, pool.used[1]);
        ebt::assert_equals(1, pool.used[3]);
        ebt::assert_equals(2, pool.used[4]);
        ebt::assert_equals(1, pool.used[7]);
        ebt::assert_equals(7, pool.block[0]);
        ebt::assert_equals(4, pool.block[2]);
        ebt::assert_equals((void *) 0, p1);
        ebt::assert_equals((void *) (2 * (1 << 10)), p2);
    }},

    {"test-alloc-one-and-free", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p1 = pool.malloc(1 << 10);

        ebt::assert_equals(1, pool.used[0]);
        ebt::assert_equals(1, pool.used[1]);
        ebt::assert_equals(1, pool.used[3]);
        ebt::assert_equals(1, pool.used[7]);
        ebt::assert_equals(7, pool.block[0]);
        ebt::assert_equals((void *) 0, p1);

        pool.free(p1);

        ebt::assert_equals(0, pool.used[0]);
        ebt::assert_equals(0, pool.used[1]);
        ebt::assert_equals(0, pool.used[3]);
        ebt::assert_equals(0, pool.used[7]);
        ebt::assert_equals(-1, pool.block[0]);
    }},

    {"test-alloc-two-and-free", []() {
        la::gpu::mem_pool pool {(void *) 0, 10, 4};

        void *p1 = pool.malloc(1 << 10);
        void *p2 = pool.malloc(2 * (1 << 10));

        ebt::assert_equals(3, pool.used[0]);
        ebt::assert_equals(3, pool.used[1]);
        ebt::assert_equals(1, pool.used[3]);
        ebt::assert_equals(2, pool.used[4]);
        ebt::assert_equals(1, pool.used[7]);
        ebt::assert_equals(7, pool.block[0]);
        ebt::assert_equals(4, pool.block[2]);
        ebt::assert_equals((void *) 0, p1);
        ebt::assert_equals((void *) (2 * (1 << 10)), p2);

        pool.free(p1);

        ebt::assert_equals(2, pool.used[0]);
        ebt::assert_equals(2, pool.used[1]);
        ebt::assert_equals(2, pool.used[4]);
        ebt::assert_equals(-1, pool.block[0]);

        ebt::assert_equals(0, pool.used[3]);
        ebt::assert_equals(0, pool.used[7]);
        ebt::assert_equals(4, pool.block[2]);
    }},

};

int main()
{
    for (auto& t: tests) {
        std::cout << t.first << std::endl;
        (t.second)();
    }

    return 0;
}
