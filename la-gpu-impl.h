
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
        vector<T>& vector<T>::operator=(vector_like<T> const& v)
        {
            if (size_ == v.size()) {
                cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                cudaFree(data_);
                cudaMalloc(&data_, v.size() * sizeof(T));
                cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyDeviceToDevice);
                size_ = v.size();
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
        matrix<T>::matrix(matrix<T> const& that)
            : data_(that.data_), rows_(that.rows_), cols_(that.cols_)
        {}

        template <class T>
        matrix<T>::matrix(matrix<T>&& that)
            : data_(std::move(that.data_)), rows_(that.rows_), cols_(that.cols_)
        {}

        template <class T>
        matrix<T>::matrix(matrix_like<T> const& m)
            : data_(), rows_(m.rows()), cols_(m.cols())
        {
            data_ = weak_vector<T>{const_cast<T*>(m.data()), m.rows() * m.cols()};
        }

        template <class T>
        matrix<T>::matrix(la::matrix_like<T> const& m)
        {
            data_.resize(m.rows() * m.cols());
            cublasSetVector(m.rows() * m.cols(), sizeof(T), m.data(), 1, data_.data(), 1);
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
        vector_like<T>& matrix<T>::as_vector()
        {
            return data_;
        }

        template <class T>
        vector_like<T> const& matrix<T>::as_vector() const
        {
            return data_;
        }

        template <class T>
        la::matrix<T> to_host(matrix_like<T> const& m)
        {
            la::matrix<T> result;
            result.resize(m.rows(), m.cols());
            cublasGetVector(m.rows() * m.cols(), sizeof(T), m.data(), 1, result.data(), 1); 
            return result;
        }

        template <class T>
        void to_device(matrix_like<T>& dm, la::matrix_like<T> const& hm)
        {
            assert(dm.rows() == hm.rows() && dm.cols() == hm.cols());

            cublasSetVector(dm.rows() * dm.cols(), sizeof(T), hm.data(), 1, dm.data(), 1);
        }

        // weak_matrix

        template <class T>
        weak_matrix<T>::weak_matrix(matrix_like<T>& m)
            : data_(m.data(), m.rows() * m.cols()), rows_(m.rows()), cols_(m.cols())
        {}

        template <class T>
        weak_matrix<T>::weak_matrix(T *data, unsigned int rows, unsigned int cols)
            : data_(data, rows * cols), rows_(rows), cols_(cols)
        {}

        template <class T>
        T* weak_matrix<T>::data()
        {
            return data_.data();
        }

        template <class T>
        T const* weak_matrix<T>::data() const
        {
            return data_.data();
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
        vector_like<T>& weak_matrix<T>::as_vector()
        {
            return data_;
        }

        template <class T>
        vector_like<T> const& weak_matrix<T>::as_vector() const
        {
            return data_;
        }

        // tensor_like

        template <class T>
        tensor_like<T>::~tensor_like()
        {}

        // tensor

        template <class T>
        tensor<T>::tensor()
            : data_(), sizes_(), dim_(0), vec_size_(0), vec_(data_.data(), 0), mat_(data_.data(), 0, 0)
        {
        }

        template <class T>
        tensor<T>::tensor(tensor<T>&& that)
            : data_(std::move(that.data_)), sizes_(std::move(that.sizes_)), dim_(that.dim_), vec_size_(that.vec_size_)
            , vec_(data_.data(), vec_size_)
            , mat_(data_.data(), vec_size_ / sizes_.back(), sizes_.back())
        {}

        template <class T>
        tensor<T>::tensor(tensor<T> const& that)
            : data_(that.data_), sizes_(that.sizes_), dim_(that.dim_), vec_size_(that.vec_size_)
            , vec_(data_.data(), vec_size_)
            , mat_(data_.data(), vec_size_ / sizes_.back(), sizes_.back())
        {}

        template <class T>
        tensor<T>::tensor(la::tensor_like<T> const& ht)
            : data_(ht.as_vector()), sizes_(ht.sizes()), dim_(ht.dim()), vec_size_(ht.vec_size())
            , vec_(data_.data(), vec_size_)
            , mat_(data_.data(), vec_size_ / sizes_.back(), sizes_.back())
        {}

        template <class T>
        tensor<T>::tensor(vector<T>&& data, std::vector<unsigned int> sizes)
            : data_(std::move(data)), sizes_(sizes), dim_(0), vec_size_(0)
            , vec_(data_.data(), 0), mat_(data_.data(), 0, 0)
        {
            dim_ = sizes_.size();

            if (dim_ != 0) {
                unsigned int d = 1;
                for (int i = 0; i < dim_; ++i) {
                    d *= sizes_[i];
                }

                vec_size_ = d;

                vec_ = weak_vector<T>{data_.data(), vec_size_};
                mat_ = weak_matrix<T>{data_.data(), d / sizes_.back(), sizes_.back()};
            } else {
                vec_size_ = 0;
                vec_ = weak_vector<T>{data_.data(), 0};
                mat_ = weak_matrix<T>{data_.data(), 0, 0};
            }
        }

        template <class T>
        tensor<T>::tensor(vector_like<T> const& data, std::vector<unsigned int> sizes)
            : tensor(vector<T>(data), sizes)
        {}

        template <class T>
        tensor<T>::tensor(vector_like<T> const& v)
            : tensor(v, {v.size()})
        {}

        template <class T>
        tensor<T>::tensor(matrix_like<T> const& m)
            : tensor(weak_vector<T>(const_cast<double*>(m.data()),
                m.rows() * m.cols()), {m.rows(), m.cols()})
        {}

        template <class T>
        T* tensor<T>::data()
        {
            return data_.data();
        }

        template <class T>
        T const* tensor<T>::data() const
        {
            return data_.data();
        }

        template <class T>
        unsigned int tensor<T>::size(unsigned int d) const
        {
            return sizes_.at(d);
        }

        template <class T>
        void tensor<T>::resize(std::vector<unsigned int> new_sizes, T value)
        {
            if (new_sizes.size() == 0) {
                sizes_ = std::vector<unsigned int>();
                data_.resize(0);
                dim_ = 0;
                vec_size_ = 0;

                vec_ = weak_vector<T>{data_.data(), 0};
                mat_ = weak_matrix<T>{data_.data(), 0, 0};
            } else {
                unsigned int d = 1;

                for (auto& s: new_sizes) {
                    d *= s;
                }

                sizes_ = new_sizes;
                data_.resize(d, value);

                dim_ = sizes_.size();
                vec_size_ = d;

                vec_ = weak_vector<T>{data_.data(), vec_size_};
                mat_ = weak_matrix<T>{data_.data(), d / sizes_.back(), sizes_.back()};
            }
        }

        template <class T>
        tensor<T>& tensor<T>::operator=(tensor<T>&& that)
        {
            data_ = std::move(that.data_);
            dim_ = that.dim_;
            sizes_ = std::move(that.sizes_);
            vec_size_ = that.vec_size_;
            vec_ = weak_vector<T>(data_.data(), vec_size_);
            mat_ = weak_matrix<T>(data_.data(), vec_size_ / sizes_.back(), sizes_.back());

            return *this;
        }

        template <class T>
        tensor<T>& tensor<T>::operator=(tensor<T> const& that)
        {
            data_ = that.data_;
            dim_ = that.dim_;
            sizes_ = that.sizes_;
            vec_size_ = that.vec_size_;
            vec_ = weak_vector<T>(data_.data(), vec_size_);
            mat_ = weak_matrix<T>(data_.data(), vec_size_ / sizes_.back(), sizes_.back());

            return *this;
        }

        template <class T>
        unsigned int tensor<T>::dim() const
        {
            return dim_;
        }

        template <class T>
        unsigned int tensor<T>::vec_size() const
        {
            return vec_size_;
        }

        template <class T>
        std::vector<unsigned int> tensor<T>::sizes() const
        {
            return sizes_;
        }

        template <class T>
        vector_like<T>& tensor<T>::as_vector()
        {
            return vec_;
        }

        template <class T>
        vector_like<T> const& tensor<T>::as_vector() const
        {
            return vec_;
        }

        template <class T>
        matrix_like<T>& tensor<T>::as_matrix()
        {
            return mat_;
        }

        template <class T>
        matrix_like<T> const& tensor<T>::as_matrix() const
        {
            return mat_;
        }

        template <class T>
        la::tensor<T> to_host(tensor_like<T> const& t)
        {
            la::vector<T> hv = to_host(t.as_vector());
            return la::tensor<T>(hv, t.sizes());
        }

        template <class T>
        void to_device(tensor_like<T>& dt, la::tensor_like<T> const& ht)
        {
            assert(dt.vec_size() == ht.vec_size());

            cublasSetVector(dt.vec_size(), sizeof(T), ht.data(), 1, dt.data(), 1);
        }

        template <class T>
        weak_tensor<T>::weak_tensor(T *data,
            std::vector<unsigned int> sizes)
            : data_(data, 0), sizes_(sizes)
            , dim_(sizes_.size())
            , mat_(data, 0, 0)
        {
            if (dim_ != 0) {
                unsigned int d = 1;
                for (int i = 0; i < dim_; ++i) {
                    d *= sizes_[i];
                }

                data_ = weak_vector<T>{data, d};
                mat_ = weak_matrix<T>{data, d / sizes_.back(), sizes_.back()};
            } else {
                data_ = weak_vector<T>{data, 0};
                mat_ = weak_matrix<T>{data, 0, 0};
            }
        }

        template <class T>
        weak_tensor<T>::weak_tensor(tensor_like<T>& t)
            : weak_tensor(t.data(), t.sizes())
        {
        }

        template <class T>
        weak_tensor<T>::weak_tensor(vector_like<T> const& v)
            : weak_tensor(const_cast<T*>(v.data()), {v.size()})
        {
        }

        template <class T>
        weak_tensor<T>::weak_tensor(matrix_like<T> const& m)
            : weak_tensor(const_cast<T*>(m.data()), {m.rows(), m.cols()})
        {
        }

        template <class T>
        T* weak_tensor<T>::data()
        {
            return data_.data();
        }

        template <class T>
        T const* weak_tensor<T>::data() const
        {
            return data_.data();
        }

        template <class T>
        unsigned int weak_tensor<T>::size(unsigned int d) const
        {
            return sizes_.at(d);
        }

        template <class T>
        unsigned int weak_tensor<T>::dim() const
        {
            return dim_;
        }

        template <class T>
        unsigned int weak_tensor<T>::vec_size() const
        {
            return data_.size();
        }

        template <class T>
        std::vector<unsigned int> weak_tensor<T>::sizes() const
        {
            return sizes_;
        }

        template <class T>
        vector_like<T>& weak_tensor<T>::as_vector()
        {
            return data_;
        }

        template <class T>
        vector_like<T> const& weak_tensor<T>::as_vector() const
        {
            return data_;
        }

        template <class T>
        matrix_like<T>& weak_tensor<T>::as_matrix()
        {
            return mat_;
        }

        template <class T>
        matrix_like<T> const& weak_tensor<T>::as_matrix() const
        {
            return mat_;
        }

    }
}
