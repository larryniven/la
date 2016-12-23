namespace la {

    // vector_like

    template <class T>
    vector_like<T>::~vector_like()
    {}

    // vector

    template <class T>
    vector<T>::vector()
    {}

    template <class T>
    vector<T>::vector(std::vector<T> const& data)
        : data_(data)
    {}

    template <class T>
    vector<T>::vector(std::vector<T>&& data)
        : data_(std::move(data))
    {}

    template <class T>
    vector<T>::vector(vector_like<T> const& v)
        : data_(v.data(), v.data() + v.size())
    {}

    template <class T>
    vector<T>::vector(std::initializer_list<T> list)
        : data_(list)
    {}

    template <class T>
    T* vector<T>::data()
    {
        return data_.data();
    }

    template <class T>
    T const* vector<T>::data() const
    {
        return data_.data();
    }

    template <class T>
    unsigned int vector<T>::size() const
    {
        return data_.size();
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
        return data_.at(i);
    }

    template <class T>
    T const& vector<T>::at(int i) const
    {
        return data_.at(i);
    }

    template <class T>
    void vector<T>::resize(unsigned int size, T value)
    {
        data_.resize(size, value);
    }

    // weak_vector

    template <class T>
    weak_vector<T>::weak_vector(vector_like<T>& data)
        : data_(data.data()), size_(data.size())
    {}

    template <class T>
    weak_vector<T>::weak_vector(T *data, unsigned int size)
        : data_(data), size_(size)
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
    T& weak_vector<T>::operator()(int i)
    {
        return data_[i];
    }

    template <class T>
    T const& weak_vector<T>::operator()(int i) const
    {
        return data_[i];
    }

    template <class T>
    T& weak_vector<T>::at(int i)
    {
        assert(i < size_);
        return data_[i];
    }

    template <class T>
    T const& weak_vector<T>::at(int i) const
    {
        assert(i < size_);
        return data_[i];
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
    matrix<T>::matrix(std::vector<std::vector<T>> data)
    {
        std::vector<T> m;

        rows_ = data.size();
        cols_ = data.front().size();

        for (auto& v: data) {
            m.insert(m.end(), v.begin(), v.end());
        }

        data_ = vector<T>(std::move(m));
    }

    template <class T>
    matrix<T>::matrix(matrix_like<T> const& m)
        : data_(std::vector<T> { m.data(), m.data() + m.rows() * m.cols() })
        , rows_(m.rows()), cols_(m.cols())
    {}

    template <class T>
    matrix<T>::matrix(std::initializer_list<std::initializer_list<T>> data)
    {
        std::vector<T> m;

        rows_ = data.size();
        cols_ = data.begin()->size();

        for (auto& v: data) {
            m.insert(m.end(), v.begin(), v.end());
        }

        data_ = vector<T>(std::move(m));
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
    T& matrix<T>::operator()(unsigned int r, unsigned int c)
    {
        return data_(r * cols_ + c);
    }

    template <class T>
    T const& matrix<T>::operator()(unsigned int r, unsigned int c) const
    {
        return data_(r * cols_ + c);
    }

    template <class T>
    T& matrix<T>::at(unsigned int r, unsigned int c)
    {
        return data_.at(r * cols_ + c);
    }

    template <class T>
    T const& matrix<T>::at(unsigned int r, unsigned int c) const
    {
        return data_.at(r * cols_ + c);
    }

    template <class T>
    void matrix<T>::resize(unsigned int rows, unsigned int cols, T value)
    {
        data_.resize(rows * cols, value);
        rows_ = rows;
        cols_ = cols;
    }

    // weak_matrix

    template <class T>
    weak_matrix<T>::weak_matrix(matrix_like<T>& data)
        : data_(data.data()), rows_(data.rows()), cols_(data.cols())
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
    T& weak_matrix<T>::operator()(unsigned int r, unsigned int c)
    {
        return data_[r * cols_ + c];
    }

    template <class T>
    T const& weak_matrix<T>::operator()(unsigned int r, unsigned int c) const
    {
        return data_[r * cols_ + c];
    }

    template <class T>
    T& weak_matrix<T>::at(unsigned int r, unsigned int c)
    {
        assert(r < rows_ && c < cols_);
        return data_[r * cols_ + c];
    }

    template <class T>
    T const& weak_matrix<T>::at(unsigned int r, unsigned int c) const
    {
        assert(r < rows_ && c < cols_);
        return data_[r * cols_ + c];
    }

    // tensor_like

    template <class T>
    tensor_like<T>::~tensor_like()
    {}

    template <class T>
    la::weak_vector<T> tensor_like<T>::as_vector() const
    {
        return la::weak_vector<T> { const_cast<T*>(data()), vec_size() };
    }

    template <class T>
    la::weak_matrix<T> tensor_like<T>::as_matrix() const
    {
        if (dim() == 0) {
            return la::weak_matrix<T> { const_cast<T*>(data()), 0, 0 };
        }

        unsigned int col = size(dim() - 1);
        unsigned int row = vec_size() / col;

        return la::weak_matrix<T> { const_cast<T*>(data()), row, col };
    }

    /*
    template <class T>
    unsigned int tensor_like<T>::vec_size() const
    {
        if (dim() == 0) {
            return 0;
        }

        unsigned int d = 1;
        for (int i = 0; i < dim(); ++i) {
            d *= size(i);
        }

        return d;
    }
    */

    template <class T>
    std::vector<unsigned int> tensor_like<T>::sizes() const
    {
        std::vector<unsigned int> result;

        for (int i = 0; i < dim(); ++i) {
            result.push_back(size(i));
        }

        return result;
    }

    // tensor

    template <class T>
    tensor<T>::tensor()
    {
    }

    template <class T>
    tensor<T>::tensor(la::vector<T> data, std::vector<unsigned int> sizes)
        : data_(data), sizes_(sizes)
    {
        dim_ = sizes_.size();

        unsigned int d = 1;
        for (int i = 0; i < dim_; ++i) {
            d *= sizes_(i);
        }

        vec_size_ = d;
    }

    template <class T>
    tensor<T>::tensor(la::vector_like<T> const& v)
        : data_(v), sizes_({v.size()})
    {
        dim_ = sizes_.size();

        unsigned int d = 1;
        for (int i = 0; i < dim_; ++i) {
            d *= sizes_(i);
        }

        vec_size_ = d;
    }

    template <class T>
    tensor<T>::tensor(la::matrix_like<T> const& m)
        : data_(la::weak_vector<T>(const_cast<double*>(m.data()), m.rows() * m.cols()))
        , sizes_({m.rows(), m.cols()})
    {
        dim_ = sizes_.size();

        unsigned int d = 1;
        for (int i = 0; i < dim_; ++i) {
            d *= sizes_(i);
        }

        vec_size_ = d;
    }

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
    T& tensor<T>::operator()(std::vector<int> indices)
    {
        assert(indices.size() == dim());

        unsigned int d = 0;

        for (int i = 0; i < indices.size(); ++i) {
            d = d * sizes_.at(i) + indices.at(i);
        }

        return data_(d);
    }

    template <class T>
    T const& tensor<T>::operator()(std::vector<int> indices) const
    {
        assert(indices.size() == dim());

        unsigned int d = indices.front();

        for (int i = 1; i < indices.size(); ++i) {
            d = d * sizes_.at(i) + indices.at(i);
        }

        return data_(d);
    }

    template <class T>
    T& tensor<T>::at(std::vector<int> indices)
    {
        assert(indices.size() == dim());

        unsigned int d = indices.front();

        for (int i = 1; i < indices.size(); ++i) {
            d = d * sizes_.at(i) + indices.at(i);
        }

        return data_.at(d);
    }

    template <class T>
    T const& tensor<T>::at(std::vector<int> indices) const
    {
        assert(indices.size() == dim());

        unsigned int d = indices.front();

        for (int i = 1; i < indices.size(); ++i) {
            d = d * sizes_.at(i) + indices.at(i);
        }

        return data_.at(d);
    }

    template <class T>
    void tensor<T>::resize(std::vector<unsigned int> sizes, T value)
    {
        if (sizes.size() == 0) {
            sizes_ = la::vector<unsigned int>();
            data_.resize(0);
            dim_ = 0;
            vec_size_ = 0;
        } else {
            unsigned int d = 1;

            for (auto& s: sizes) {
                d *= s;
            }

            sizes_ = la::vector<unsigned int>(sizes);
            data_.resize(d, value);

            dim_ = sizes_.size();
            vec_size_ = d;
        }
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

    // weak_tensor

    template <class T>
    weak_tensor<T>::weak_tensor(tensor_like<T>& t)
        : data_(t.data())
    {
        sizes_.resize(t.dim());

        unsigned int d = 1;
        for (int i = 0; i < t.dim(); ++i) {
            d *= t.size(i);
            sizes_(i) = t.size(i);
        }

        vec_size_ = d;

        dim_ = sizes_.size();
    }

    template <class T>
    weak_tensor<T>::weak_tensor(T *data,
        std::vector<unsigned int> sizes)
        : data_(data), sizes_(sizes)
    {
        dim_ = sizes_.size();

        unsigned int d = 1;
        for (int i = 0; i < dim_; ++i) {
            d *= sizes_(i);
        }

        vec_size_ = d;
    }

    template <class T>
    weak_tensor<T>::weak_tensor(la::vector_like<T> const& v)
        : data_(const_cast<T*>(v.data()))
        , sizes_({v.size()})
    {
        dim_ = sizes_.size();

        unsigned int d = 1;
        for (int i = 0; i < dim_; ++i) {
            d *= sizes_(i);
        }

        vec_size_ = d;
    }

    template <class T>
    weak_tensor<T>::weak_tensor(la::matrix_like<T> const& m)
        : data_(const_cast<T*>(m.data()))
        , sizes_({m.rows(), m.cols()})
    {
        dim_ = sizes_.size();

        unsigned int d = 1;
        for (int i = 0; i < dim_; ++i) {
            d *= sizes_(i);
        }

        vec_size_ = d;
    }

    template <class T>
    T* weak_tensor<T>::data()
    {
        return data_;
    }

    template <class T>
    T const* weak_tensor<T>::data() const
    {
        return data_;
    }

    template <class T>
    unsigned int weak_tensor<T>::size(unsigned int d) const
    {
        return sizes_(d);
    }

    template <class T>
    T& weak_tensor<T>::operator()(std::vector<int> indices)
    {
        assert(indices.size() == dim());

        unsigned int d = indices.front();

        for (int i = 1; i < indices.size(); ++i) {
            d = d * sizes_(i) + indices.at(i);
        }

        return data_[d];
    }

    template <class T>
    T const& weak_tensor<T>::operator()(std::vector<int> indices) const
    {
        assert(indices.size() == dim());

        unsigned int d = indices.front();

        for (int i = 1; i < indices.size(); ++i) {
            d = d * sizes_(i) + indices.at(i);
        }

        return data_[d];
    }

    template <class T>
    T& weak_tensor<T>::at(std::vector<int> indices)
    {
        assert(indices.size() == dim());

        unsigned int d = indices.front();

        for (int i = 1; i < indices.size(); ++i) {
            d = d * sizes_(i) + indices.at(i);
        }

        return data_[d];
    }

    template <class T>
    T const& weak_tensor<T>::at(std::vector<int> indices) const
    {
        assert(indices.size() == dim());

        unsigned int d = indices.front();

        for (int i = 1; i < indices.size(); ++i) {
            d = d * sizes_(i) + indices.at(i);
        }

        return data_[d];
    }

    template <class T>
    unsigned int weak_tensor<T>::dim() const
    {
        return dim_;
    }

    template <class T>
    unsigned int weak_tensor<T>::vec_size() const
    {
        return vec_size_;
    }

    template <class T>
    matrix<T> trans(matrix_like<T> const& m)
    {
        matrix<T> result;
        result.resize(m.cols(), m.rows());

        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                result(j, i) = m(i, j);
            }
        }

        return result;
    }

}

namespace ebt {
    namespace json {
    
        template <class T>
        la::vector<T> json_parser<la::vector<T>>::parse(std::istream& is)
        {
            json_parser<std::vector<double>> vec_parser;
            return la::vector<T>(vec_parser.parse(is));
        }
    
        template <class T>
        la::matrix<T> json_parser<la::matrix<T>>::parse(std::istream& is)
        {
            json_parser<std::vector<std::vector<double>>> mat_parser;
            return la::matrix<T>(mat_parser.parse(is));
        }

        template <class T>
        la::tensor<T> json_parser<la::tensor<T>>::parse(std::istream& is)
        {
            json_parser<std::tuple<std::vector<int>, la::vector<T>>> parser;
            std::tuple<std::vector<int>, la::vector<T>> data = parser.parse(is);

            std::vector<unsigned int> sizes;

            for (auto& i: std::get<0>(data)) {
                sizes.push_back((unsigned int)(i));
            }

            return la::tensor<T>(std::move(std::get<1>(data)), sizes);
        }

        template <class T>
        void json_writer<la::vector<T>>::write(la::vector<T> const& v, std::ostream& os)
        {
            std::vector<T> vec {v.data(), v.data() + v.size()};
            ebt::json::json_writer<std::vector<T>> writer;
            writer.write(vec, os);
        }
    
        template <class T>
        void json_writer<la::matrix<T>>::write(la::matrix<T> const& m, std::ostream& os)
        {
            std::vector<std::vector<T>> mat;

            for (T const *i = m.data(); i < m.data() + m.rows() * m.cols(); i = i + m.cols()) {
                mat.push_back(std::vector<T> {i, i + m.cols()});
            }

            ebt::json::json_writer<std::vector<std::vector<T>>> writer;
            writer.write(mat, os);
        }

        template <class T>
        void json_writer<la::tensor<T>>::write(la::tensor<T> const& t, std::ostream& os)
        {
            std::tuple<std::vector<int>, la::vector<T>> data;

            unsigned int dim = 1;

            for (int i = 0; i < t.dim(); ++i) {
                std::get<0>(data).push_back(t.size(i));

                dim *= t.size(i);
            }

            std::get<1>(data) = la::vector<T>(la::weak_vector<T>(const_cast<T*>(t.data()), dim));

            ebt::json::json_writer<std::tuple<std::vector<int>, la::vector<T>>> writer;
            writer.write(data, os);
        }
    }
}
