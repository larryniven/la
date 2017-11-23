#ifndef LA_H
#define LA_H

#include <vector>

namespace la {

    template <class T>
    struct vector_like {

        virtual ~vector_like()
        {}

        virtual T* data() = 0;
        virtual T const* data() const = 0;

        virtual unsigned int size() const = 0;

    };

    template <class T>
    struct matrix_like {

        virtual ~matrix_like()
        {}

        virtual T* data() = 0;
        virtual T const* data() const = 0;

        virtual unsigned int rows() const = 0;
        virtual unsigned int cols() const = 0;

    };

    template <class T>
    struct tensor_like {

        virtual ~tensor_like()
        {}

        virtual T* data() = 0;
        virtual T const* data() const = 0;

        virtual unsigned int dim() const = 0;
        virtual unsigned int size(unsigned int d) const = 0;

        virtual unsigned int vec_size() const = 0;
        virtual std::vector<unsigned int> sizes() const = 0;

    };

}

#endif
