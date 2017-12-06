#include "la/mem-pool.h"
#include <cassert>
#include <iostream>

namespace la {

    namespace gpu {

        mem_pool::mem_pool(void *dev_ptr, unsigned int base_power, unsigned int total_depth)
            : dev_ptr((char*) dev_ptr), base_power(base_power), total_depth(total_depth)
        {
            base_size = (1 << base_power);

            // total blocks = 2^0 + 2^1 + ... + 2^(total_depth-1)
            used.resize((1 << total_depth) - 1);

            block.resize((1 << (total_depth - 1)), -1);
        }

        void* mem_pool::malloc(size_t size)
        {
            unsigned int req = (size >> base_power) + (size % base_size > 0 ? 1 : 0);

            unsigned int block_size = (1 << (total_depth - 1));
            unsigned int num_blocks = 1;
            unsigned int depth = 0;

            // find block k

            unsigned int k = 0;

            while (k < used.size()) {
                assert(k < (num_blocks << 1) - 1);

                if (req <= block_size - used[k] && req > (block_size >> 1)) {
                    break;
                } else if (req < block_size) {
                    k = left_child(k);
                    block_size = (block_size >> 1);
                    ++depth;
                    num_blocks = (num_blocks << 1);
                } else {
                    ++k;
                }
            }

            if (k == used.size()) {
                throw std::logic_error("not enough memory");
            }

            // update parent

            unsigned int k_up = k;

            while (k_up != 0) {
                used[k_up] += block_size;
                k_up = parent(k_up);
            }
            used[k_up] += block_size;

            // compute result

            unsigned int shift = (k - num_blocks + 1) << (total_depth - depth - 1);

            assert(block[shift] == -1);
            block[shift] = k;

            void *result = dev_ptr + (shift << base_power);

            return result;
        }

        void mem_pool::free(void *p)
        {
            unsigned int shift = (((char*)p) - dev_ptr) >> base_power;

            unsigned int k = block[shift];

            unsigned int block_size = (1 << (total_depth - 1));
            unsigned int num_blocks = 1;

            while (k >= (num_blocks << 1) - 1) {
                num_blocks = (num_blocks << 1);
                block_size = (block_size >> 1);
            }

            while (k != 0) {
                used[k] -= block_size;
                k = parent(k);
            }

            used[k] -= block_size;
            block[shift] = -1;
        }

        int mem_pool::left_child(int k)
        {
            return (k << 1) + 1;
        }

        int mem_pool::right_child(int k)
        {
            return (k << 1) + 2;
        }

        int mem_pool::parent(int k)
        {
            // note the integer div

            return (k - 1) / 2;
        }

    }

}
