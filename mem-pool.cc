#include "la/mem-pool.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace la {

    namespace gpu {

        mem_pool::mem_pool(void *dev_ptr, unsigned int base_power, unsigned int total_depth)
            : dev_ptr((char*) dev_ptr), base_power(base_power), total_depth(total_depth)
        {
            base_size = (1L << base_power);

            // total blocks = 2^0 + 2^1 + ... + 2^(total_depth-1) = 2^total_depth - 1
            lost.resize((1L << total_depth) - 1);

            block.resize((1L << (total_depth - 1)), -1);
        }

        void* mem_pool::malloc(size_t size)
        {
            long req = (size >> base_power) + (size % base_size > 0 ? 1 : 0);

            size_t block_size = (1L << (total_depth - 1));
            long num_blocks = 1;
            unsigned int depth = 0;

            // find block k

            long k = 0;

            // (num_blocks << 1) - 1 is the first block of the next layer
            while (k < (num_blocks << 1) - 1) {
                if (req <= block_size && req > (block_size >> 1) && lost[k] == 0) {
                    break;
                } else if (req < block_size && depth < total_depth - 1
                        && req <= (block_size >> lost[k])) {
                    k = left_child(k);
                    block_size = (block_size >> 1);
                    num_blocks = (num_blocks << 1);
                    ++depth;
                } else {
                    ++k;
                }
            }

            if (k == (num_blocks << 1) - 1) {
                std::ostringstream oss;
                oss << "hitting the boundary k: " << k << " # blocks: "  << num_blocks << std::endl;
                throw std::logic_error(oss.str());
            }

            assert(lost[k] == 0);
            lost[k] = total_depth - depth;

            // update parent

            update_parent(k, depth);

            // compute result

            unsigned int shift = (k - num_blocks + 1) << (total_depth - depth - 1);
            assert(shift < block.size());
            assert(block[shift] == -1);
            block[shift] = k;

            void *result = dev_ptr + (shift << base_power);

            return result;
        }

        void mem_pool::free(void *p)
        {
            if (p == (void *) 0) {
                return;
            }

            unsigned int shift = (((char*)p) - dev_ptr) >> base_power;

            assert(shift < block.size());
            assert(block[shift] != -1);

            long k = block[shift];

            size_t block_size = (1 << (total_depth - 1));
            long num_blocks = 1;
            unsigned int depth = 0;

            while (k >= (num_blocks << 1) - 1) {
                num_blocks = (num_blocks << 1);
                block_size = (block_size >> 1);
                ++depth;
            }

            assert(lost[k] == total_depth - depth);
            lost[k] = 0;

            update_parent(k, depth);

            block[shift] = -1;
        }

        void mem_pool::update_parent(long k, unsigned int depth)
        {
            long k_up = k;
            unsigned int depth_up = depth;

            while (k_up > 0) {
                k_up = parent(k_up);
                --depth_up;

                assert(lost[k_up] <= total_depth - depth_up);

                unsigned int left_lost = lost[left_child(k_up)];
                unsigned int right_lost = lost[right_child(k_up)];

                unsigned int old_lost = lost[k_up];

                if (left_lost == 0 && right_lost == 0) {
                    lost[k_up] = 0;
                } else {
                    unsigned int cand = std::min(left_lost, right_lost) + 1;
                    lost[k_up] = cand;
                }

                if (old_lost == lost[k_up]) {
                    break;
                }
            }
        }

        long mem_pool::left_child(long k)
        {
            return (k << 1) + 1;
        }

        long mem_pool::right_child(long k)
        {
            return (k << 1) + 2;
        }

        long mem_pool::parent(long k)
        {
            // note the integer div

            return (k - 1) / 2;
        }

    }

}
