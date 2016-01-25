namespace la {

    template <class T>
    vector<T>::vector()
    {}

    template <class T>
    vector<T>::vector(std::vector<T> v)
        : vec_(std::move(v))
    {}

    template <class T>
    vector<T>::vector(std::initializer_list<T> const& list)
        : vec_(list)
    {}

    template <class T>
    T* vector<T>::data()
    {
        return vec_.data();
    }

    template <class T>
    T const* vector<T>::data() const
    {
        return vec_.data();
    }

    template <class T>
    unsigned int vector<T>::size() const
    {
        return vec_.size();
    }

    template <class T>
    void vector<T>::resize(unsigned int size, T value)
    {
        return vec_.resize(size, value);
    }

    template <class T>
    T& vector<T>::operator()(int i)
    {
        return vec_[i];
    }

    template <class T>
    T const& vector<T>::operator()(int i) const
    {
        return vec_[i];
    }

    template <class T>
    T& vector<T>::at(int i)
    {
        return vec_.at(i);
    }

    template <class T>
    T const& vector<T>::at(int i) const
    {
        return vec_.at(i);
    }

    template <class T>
    matrix<T>::matrix()
    {}

    template <class T>
    matrix<T>::matrix(std::vector<std::vector<T>> const& m)
    {
        rows_ = m.size();
        cols_ = m.front().size();
        for (auto& v: m) {
            vec_.insert(vec_.end(), v.begin(), v.end());
        }
    }

    template <class T>
    matrix<T>::matrix(std::initializer_list<std::initializer_list<T>> const& list)
    {
        rows_ = list.size();
        cols_ = list.begin()->size();
        for (auto& ell: list) {
            vec_.insert(vec_.end(), ell.begin(), ell.end());
        }
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
        vec_.resize(rows * cols, value);
        rows_ = rows;
        cols_ = cols;
    }

    template <class T>
    T& matrix<T>::operator()(unsigned int r, unsigned int c)
    {
        return vec_[r * cols_ + c];
    }

    template <class T>
    T const& matrix<T>::operator()(unsigned int r, unsigned int c) const
    {
        return vec_[r * cols_ + c];
    }

    template <class T>
    T& matrix<T>::at(unsigned int r, unsigned int c)
    {
        return vec_.at(r * cols_ + c);
    }

    template <class T>
    T const& matrix<T>::at(unsigned int r, unsigned int c) const
    {
        return vec_.at(r * cols_ + c);
    }

    template <class T>
    matrix<T> trans(matrix<T> const& m)
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

    }
}
