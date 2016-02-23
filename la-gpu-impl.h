
namespace la {

    namespace gpu {

        // vector_like

        template <class T>
        vector_like<T>::~vector_like()
        {}

        // vector

        template <class T>
        vector<T>::vector()
            : data_(nullptr), size_(0)
        {}

        template <class T>
        vector<T>::vector(vector_like<T> const& v)
        {
            cudaMalloc(&data_, v.size() * sizeof(T));
            cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyDeviceToDevice);
            size_ = v.size();
        }

        template <class T>
        vector<T>::vector(la::vector_like<T> const& v)
        {
            cudaMalloc(&data_, v.size() * sizeof(T));
            cublasSetVector(v.size(), sizeof(T), v.data(), 1, data_, 1);
            size_ = v.size();
        }

        template <class T>
        vector<T>::~vector()
        {
            cudaFree(data_);
        }

        template <class T>
        vector<T>::vector(vector<T> const& v)
        {
            cudaMalloc(&data_, v.size() * sizeof(T));
            cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyDeviceToDevice);
            size_ = v.size_;
        }

        template <class T>
        vector<T>::vector(vector<T>&& v)
        {
            data_ = v.data_;
            size_ = v.size_;
            v.data_ = nullptr;
            v.size_ = 0;
        }

        template <class T>
        vector<T>& vector<T>::operator=(vector<T> const& v)
        {
            if (size_ == v.size_) {
                cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                cudaFree(data_);
                cudaMalloc(&data_, v.size() * sizeof(T));
                cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyDeviceToDevice);
                size_ = v.size_;
            }

            return *this;
        }

        template <class T>
        vector<T>& vector<T>::operator=(vector<T>&& v)
        {
            using std::swap;

            swap(data_, v.data_);
            swap(size_, v.size_);

            return *this;
        }

        template <class T>
        T* vector<T>::data()
        {
            return data_;
        }

        template <class T>
        T const* vector<T>::data() const
        {
            return data_;
        }

        template <class T>
        unsigned int vector<T>::size() const
        {
            return size_;
        }

        template <class T>
        T* vector<T>::begin()
        {
            return data_;
        }

        template <class T>
        T const* vector<T>::begin() const
        {
            return data_;
        }

        template <class T>
        T* vector<T>::end()
        {
            return data_ + size_;
        }

        template <class T>
        T const* vector<T>::end() const
        {
            return data_ + size_;
        }

        template <class T>
        void vector<T>::resize(unsigned int size, T value)
        {
            if (size == size_) {
                return;
            }

            cudaFree(data_);
            cudaMalloc(&data_, size * sizeof(T));
            std::vector<T> v;
            v.resize(size, value);
            cublasSetVector(size, sizeof(T), v.data(), 1, data_, 1);
            size_ = size;
        }

        template <class T>
        la::vector<T> to_host(vector_like<T> const& v)
        {
            la::vector<T> result;
            result.resize(v.size());
            cublasGetVector(v.size(), sizeof(T), v.data(), 1, result.data(), 1);
            return result;
        }

        template <class T>
        void to_device(vector_like<T>& dv, la::vector_like<T> const& hv)
        {
            assert(dv.size() == hv.size());

            cublasSetVector(hv.size(), sizeof(T), hv.data(), 1, dv.data(), 1);
        }

        // weak_vector

        template <class T>
        weak_vector<T>::weak_vector(T *data, unsigned int size)
            : data_(data), size_(size)
        {}

        template <class T>
        weak_vector<T>::weak_vector(vector_like<T>& data)
            : data_(data.data()), size_(data.size())
        {}

        template <class T>
        T* weak_vector<T>::data()
        {
            return data_;
        }

        template <class T>
        T const* weak_vector<T>::data() const
        {
            return data_;
        }

        template <class T>
        unsigned int weak_vector<T>::size() const
        {
            return size_;
        }

        template <class T>
        T* weak_vector<T>::begin()
        {
            return data_;
        }

        template <class T>
        T const* weak_vector<T>::begin() const
        {
            return data_;
        }

        template <class T>
        T* weak_vector<T>::end()
        {
            return data_ + size_;
        }

        template <class T>
        T const* weak_vector<T>::end() const
        {
            return data_ + size_;
        }

        // matrix_like

        template <class T>
        matrix_like<T>::~matrix_like()
        {}

        // matrix

        template <class T>
        matrix<T>::matrix()
        {}

        template <class T>
        matrix<T>::matrix(matrix_like<T> const& m)
            : data_(m.data_), rows_(m.rows_), cols_(m.cols_)
        {}

        template <class T>
        matrix<T>::matrix(la::matrix_like<T> const& m)
        {
            la::matrix<T> mT = la::trans(m);
            data_.resize(m.rows() * m.cols());
            cublasSetVector(m.rows() * m.cols(), sizeof(T), mT.data(), 1, data_.data(), 1);
            rows_ = m.rows();
            cols_ = m.cols();
        }

        template <class T>
        T* matrix<T>::data()
        {
            return data_.data();
        }

        template <class T>
        T const* matrix<T>::data() const
        {
            return data_.data();
        }

        template <class T>
        unsigned int matrix<T>::rows() const
        {
            return rows_;
        }

        template <class T>
        unsigned int matrix<T>::cols() const
        {
            return cols_;
        }

        template <class T>
        void matrix<T>::resize(unsigned int rows, unsigned int cols, T value)
        {
            data_.resize(rows * cols, value);
            rows_ = rows;
            cols_ = cols;
        }

        template <class T>
        la::matrix<T> to_host(matrix_like<T> const& m)
        {
            la::matrix<T> result;
            result.resize(m.cols(), m.rows());
            cublasGetMatrix(m.rows(), m.cols(), sizeof(T), m.data(), m.rows(), result.data(), m.rows()); 
            return la::trans(result);
        }

        template <class T>
        void to_device(matrix_like<T>& dm, la::matrix_like<T> const& hm)
        {
            assert(dm.rows() == hm.rows() && dm.cols() == dm.cols());

            la::matrix<T> mT = la::trans(hm);
            cublasSetMatrix(dm.rows(), dm.cols(), sizeof(T), mT.data(), dm.rows(), dm.data(), dm.rows());
        }

        // weak_matrix

        template <class T>
        weak_matrix<T>::weak_matrix(matrix_like<T>& m)
            : data_(m.data()), rows_(m.rows()), cols_(m.cols())
        {}

        template <class T>
        weak_matrix<T>::weak_matrix(T *data, unsigned int rows, unsigned int cols)
            : data_(data), rows_(rows), cols_(cols)
        {}

        template <class T>
        T* weak_matrix<T>::data()
        {
            return data_;
        }

        template <class T>
        T const* weak_matrix<T>::data() const
        {
            return data_;
        }

        template <class T>
        unsigned int weak_matrix<T>::rows() const
        {
            return rows_;
        }

        template <class T>
        unsigned int weak_matrix<T>::cols() const
        {
            return cols_;
        }

        template <class T>
        __host__ __device__
        void idiv_op::operator()(T t) const
        {
            thrust::get<0>(t) /= thrust::get<1>(t);
        }

    }
}
