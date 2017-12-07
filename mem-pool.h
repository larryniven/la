#ifndef LA_MEM_POOL_H
#define LA_MEM_POOL_H

#include <cstring>
#include <vector>

namespace la {

    namespace gpu {

        struct mem_pool {

            unsigned int base_power;
            unsigned int total_depth;

            size_t base_size;
            char *dev_ptr;

            std::vector<unsigned int> lost;
            std::vector<long> block;

            mem_pool(void *dev_ptr, unsigned int base_power, unsigned int total_depth);

            void status();
            void* malloc(size_t size);
            void free(void *p);

            void update_parent(long k, unsigned int depth);
            long left_child(long k);
            long right_child(long k);
            long parent(long k);

        };

    }

}

#endif
