#ifndef LA_MEM_POOL_H
#define LA_MEM_POOL_H

#include <vector>

namespace la {

    namespace gpu {

        struct mem_pool {

            unsigned int base_power;
            unsigned int total_depth;

            size_t base_size;
            char *dev_ptr;

            std::vector<unsigned int> used;
            std::vector<int> block;

            mem_pool(void *dev_ptr, unsigned int base_power, unsigned int total_depth);

            void* malloc(size_t size);
            void free(void *p);

            int left_child(int k);
            int right_child(int k);
            int parent(int k);

        };

    }

}

#endif
