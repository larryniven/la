
namespace la {

    namespace gpu {

        template <class T>
        vector<T>::vector()
            : data_(nullptr), size_(0)
        {}

        template <class T>
        vector<T>::vector(la::vector<T> const& v)
        {
             cudaMalloc(&data_, sizeof(T) * v.size());
             cublasSetVector(v.size(), sizeof(T), v.data(), 1, data_, 1);
             size_ = v.size();
        }

        template <class T>
        vector<T>::vector(vector<T>&& v)
        {
             data_ = v.data_;
             v.data_ = nullptr;
             size_ = v.size_;
        }

        template <class T>
        vector<T>::vector(vector<T> const& v)
        {
            cudaMalloc(&data_, sizeof(T) * v.size());
            cudaMemcpy(data_, v.data(), sizeof(T) * v.size(), cudaMemcpyDeviceToDevice);
            size_ = v.size();
        }

        template <class T>
        vector<T>& vector<T>::operator=(vector<T> const& v)
        {
            cudaFree(data_);
            cudaMalloc(&data_, sizeof(T) * v.size());
            cudaMemcpy(data_, v.data(), sizeof(T) * v.size(), cudaMemcpyDeviceToDevice);
            size_ = v.size();
            return *this;
        }

        template <class T>
        vector<T>& vector<T>::operator=(vector<T>&& v)
        {
            data_ = v.data_;
            v.data_ = nullptr;
            size_ = v.size_;

            return *this;
        }

        template <class T>
        vector<T>::~vector()
        {
            cudaFree(data_);
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
        void vector<T>::resize(unsigned int size, T value)
        {
            cudaFree(data_);
            cudaMalloc(&data_, size * sizeof(T));
            std::vector<T> v;
            v.resize(size, value);
            cublasSetVector(size, sizeof(T), v.data(), 1, data_, 1);
            size_ = size;
        }

        template <class T>
        T& vector<T>::operator()(int i)
        {
            return data_[i];
        }

        template <class T>
        T const& vector<T>::operator()(int i) const
        {
            return data_[i];
        }

        template <class T>
        T& vector<T>::at(int i)
        {
            assert(i < size_);
            return data_[i];
        }

        template <class T>
        T const& vector<T>::at(int i) const
        {
            assert(i < size_);
            return data_[i];
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
        matrix<T>::matrix()
        {}

        template <class T>
        matrix<T>::matrix(la::matrix<T> const& m)
        {
            rows_ = m.rows();
            cols_ = m.cols();
            la::matrix<T> mT = la::trans(m);
            vec_.resize(cols_ * rows_);
            cublasSetMatrix(rows_, cols_, sizeof(T), mT.data(), rows_, vec_.data(), rows_);
        }

        template <class T>
        T* matrix<T>::data()
        {
            return vec_.data();
        }

        template <class T>
        T const* matrix<T>::data() const
        {
            return vec_.data();
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
        void matrix<T>::resize(int rows, int cols, T value)
        {
            vec_.resize(cols * rows, value);
            rows_ = rows;
            cols_ = cols;
        }

        template <class T>
        T& matrix<T>::operator()(unsigned int r, unsigned int c)
        {
            return vec_(c * rows_ + r);
        }

        template <class T>
        T const& matrix<T>::operator()(unsigned int r, unsigned int c) const
        {
            return vec_(c * rows_ + r);
        }

        template <class T>
        T& matrix<T>::at(unsigned int r, unsigned int c)
        {
            return vec_.at(c * rows_ + r);
        }

        template <class T>
        T const& matrix<T>::at(unsigned int r, unsigned int c) const
        {
            return vec_.at(c * rows_ + r);
        }

        template <class T>
        la::vector<T> to_host(vector<T> const& v)
        {
            la::vector<T> result;
            result.resize(v.size());
            cublasGetVector(v.size(), sizeof(T), v.data(), 1, result.data(), 1);
            return result;
        }

        template <class T>
        la::matrix<T> to_host(matrix<T> const& m)
        {
            la::matrix<T> result;
            result.resize(m.cols(), m.rows());
            cublasGetMatrix(m.rows(), m.cols(), sizeof(T), m.data(), m.rows(), result.data(), m.rows()); 
            return la::trans(result);
        }

        template <class T>
        __host__ __device__
        void imul_op::operator()(T t) const
        {
            thrust::get<0>(t) *= thrust::get<1>(t);
        }

        template <class T>
        __host__ __device__
        void idiv_op::operator()(T t) const
        {
            thrust::get<0>(t) /= thrust::get<1>(t);
        }

        template <class T>
        __host__ __device__
        void ilogistic_op::operator()(T t) const
        {
            thrust::get<0>(t) = 1 / (1 + std::exp(-thrust::get<1>(t)));
        }

    }
}
