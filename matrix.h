#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include <vector>
#include <iostream>
#include <iterator>
#include <cmath>
#include <thread>
#include <future>

template<typename T>
class Matrix{
public:
    int n = 0;
    int m = 0;
    std::vector<std::vector<T> > mtr;

    Matrix(int, int);
    Matrix();

    void print() const; //вывод в std::cout
    void read(); //ввод и размера, и значений из std::cin
    void read_values(); //ввод только значений из std::cin
    void fill(T); //заполнение ячеек матрицы
    static Matrix create_zero_matrix(int);
    static Matrix create_identity_matrix(int);

    void transpose(); //транспонирование матрицы
    double get_det(const Matrix& = Matrix<T>::create_zero_matrix(0)) const; //определитель матрицы
    double get_minor(int, int) const; //минор

    Matrix<T> operator!() const; //обратная матрица

    bool operator==(const Matrix&) const; //сравнение матриц
    Matrix<T>& operator=(const Matrix &); //присваивание

    void operator+=(const Matrix&); //сумма двух матрицы
    Matrix<T> operator+(const Matrix&) const;

    void operator-=(const Matrix&); //разность двух матриц
    Matrix<T> operator-(const Matrix&) const;

    void operator*=(const Matrix&); //произведение двух матриц
    Matrix<T> operator*(const Matrix&) const;

    void operator*=(int const&); //умножение матрицы на скаляр
    Matrix<T> operator*(int const&) const;

    void operator/=(int const&); //деление матрицы на скаляр
    Matrix<T> operator/(int const&) const;


    friend Matrix<T> operator*(const int& x, const Matrix& m1) {
        Matrix<T> ans = m1;
        for (int i = 0; i < ans.n; ++i) {
            for (int j = 0; j < ans.m; ++j) {
                ans.mtr[i][j] = x * m1.mtr[i][j];
            }
        }
        return ans;
    }

    friend std::ostream& operator<<(std::ostream& o_stream, const Matrix& x) {  //вывод размеров и ячеек матрицы
        o_stream << "SIZE OF MATRIX: " << x.n << ' ' << x.m << '\n';
        o_stream << "MATRIX: " << '\n';
        for (int i = 0; i < x.n; ++i) {
            for (int j = 0; j < x.m; ++j) {
                o_stream << x.mtr[i][j] << ' ';
            }
            o_stream << '\n';
        }
        return o_stream;
    }

    friend std::istream& operator>>(std::istream& i_stream, Matrix& x) { //ввод размеров и ячеек матрицы
        i_stream >> x.n;
        i_stream >> x.m;
        x.mtr.resize(x.n, std::vector<T> (x.m));
        for (int i = 0; i < x.n; ++i) {
            for (int j = 0; j < x.m; ++j) {
                i_stream >> x.mtr[i][j];
            }
        }
        return i_stream;
    }

    Matrix<T> parallel_multiply(int, int) const; //параллельное умножение матрицы на скаляр
    Matrix<T> parallel_division(int, int) const; //параллельное деление матрицы на скаляр

    Matrix<T> parallel_sum(const Matrix<T> &, int) const; //параллельная сумма матриц
    Matrix<T> parallel_diff(const Matrix<T> &, int) const; //параллельная разность матриц
    Matrix<T> parallel_mult(const Matrix<T> &, int) const; //параллельное умножение матриц

    Matrix<T> async_sum(const Matrix<T> &, int) const; //асинхронная сумма матриц
    Matrix<T> async_diff(const Matrix<T> &, int) const; //асинхронная разность матриц

    Matrix<T> async_multiply(int, int) const; //асинхронное умножение матрицы на скаляр
    Matrix<T> async_division(int, int) const; //асинхронное деление матрицы на скаляр
};


template <typename T>
Matrix<T> Matrix<T>::create_zero_matrix(int size) {
    Matrix m(size, size);
    return m;
}

template <typename T>
Matrix<T> Matrix<T>::create_identity_matrix(int size) {
    Matrix m(size, size);
    m.fill(1);
    return m;
}

template <typename T>
Matrix<T>::Matrix(int a, int b) : n(a), m(b){
    mtr.resize(n, std::vector<T> (m));
}

template <typename T>
Matrix<T>::Matrix() : Matrix(0, 0){}

template <typename T>
void Matrix<T>::print() const {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << mtr[i][j] << ' ';
        }
        std::cout << '\n';
    }
}

template <typename T>
void Matrix<T>::read() {
    std::cin >> n;
    std::cin >> m;
    mtr.resize(n, std::vector<T> (m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cin >> mtr[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::read_values() {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cin >> mtr[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::fill(T val) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            mtr[i][j] = val;
        }
    }
}

template <typename T>
void Matrix<T>::transpose() {
    if (n != m){
        throw std::invalid_argument("this matrix is not square");
    }
    T tmp;
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < m; ++j) {
            std::swap(mtr[i][j], mtr[j][i]);
        }
    }
}

template<typename T>
double Matrix<T>::get_det(const Matrix<T>& x) const{
    if (x.n == 0) {
        return get_det(*this);
    }
    if (x.n == 1) {
        return x.mtr[0][0];
    }
    if (x.n == 2) {
        return x.mtr[0][0] * x.mtr[1][1] - x.mtr[0][1] * x.mtr[1][0];
    }
    else {
        double d = 0;
        for (int k = 0; k < x.n; ++k) {
            Matrix<T> new_matrix(x.n - 1, x.m - 1);
            for (int i = 1; i < x.n; ++i) {
                int t = 0;
                for (int j = 0; j < x.m; ++j) {
                    if (j == k) {
                        continue;
                    }
                    new_matrix.mtr[i - 1][t] =  x.mtr[i][j];
                    ++t;
                }
            }
            d += std::pow(-1, k + 2) * x.mtr[0][k] * get_det(new_matrix);
        }
        return d;
    }
}

template<typename T>
double Matrix<T>::get_minor(int r, int c) const{
    Matrix<T> new_matrix(n - 1, m - 1);
    int t = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (i == r || j == c) {
                continue;
            }
            new_matrix.matrix[t / (m - 1)][t % (m - 1)] = mtr[i][j];
            ++t;
        }
    }
    return std::pow(-1, r + c) * get_det(new_matrix);
}

template <typename T>
Matrix<T> Matrix<T>::operator!() const {
    double det = get_det();
    if (!det) throw std::runtime_error("Matrix cannot be reversed.");

    Matrix new_mtr(n, m);
    Matrix<T>::transpose();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double minor = get_minor(i, j);
            new_mtr.matrix[i][j] = minor;
        }
    }

    Matrix<T>::transpose();
    return new_mtr / det;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix& x) const {
    if (n != x.n || m != x.m) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (mtr[i][j] != x.mtr[i][j])
                return false;
        }
    }
    return true;
}

template <typename T>
void Matrix<T>::operator+=(const Matrix & x) {
    if (n != x.n || m != x.m) {
        std::cerr << "different sizes of matrices!\n";
        exit(1);
    }
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                mtr[i][j] += x.mtr[i][j];
            }
        }
    }
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix & x) const {
    if (n != x.n || m != x.m) {
        std::cerr << "incorrect sizes of matrices!\n";
        exit(1);
    }
    else {
        Matrix<T> ans(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                ans.mtr[i][j] = mtr[i][j] + x.mtr[i][j];
            }
        }
        return ans;
    }
}

template <typename T>
void Matrix<T>::operator-=(const Matrix & x) {
    if (n != x.n || m != x.m) {
        std::cerr << "different sizes of matrices!\n";
        exit(1);
    }
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                mtr[i][j] -= x.mtr[i][j];
            }
        }
    }
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix & x) const {
    if (n != x.n || m != x.m) {
        std::cerr << "incorrect sizes of matrices!\n";
        exit(1);
    }
    else {
        Matrix<T> ans(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                ans.mtr[i][j] = mtr[i][j] - x.mtr[i][j];
            }
        }
        return ans;
    }
}

template <typename T>
void Matrix<T>::operator*=(const Matrix & x) {
    Matrix<T> result(std::min(n, x.n), std::min(m, x.m));
    if (m != x.n) {
        std::cerr << "incorrect sizes of matrices!";
        exit(1);
    }
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < x.m; ++j) {
                T sum = 0;
                for (int k = 0; k < x.n; ++k) {
                    sum += mtr[i][k] * x.mtr[k][j];
                }
                result.mtr[i][j] = sum;
            }
        }
        n = result.n;
        m = result.m;
        mtr = result.mtr;
    }
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix & x) const {
    Matrix<T> result(std::min(n, x.n), std::min(m, x.m));
    if (m != x.n) {
        std::cerr << "incorrect sizes of matrices!";
        exit(1);
    }
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < x.m; ++j) {
                T sum = 0;
                for (int k = 0; k < x.n; ++k) {
                    sum += mtr[i][k] * x.mtr[k][j];
                }
                result.mtr[i][j] = sum;
            }
        }
        return result;
    }
}


template <typename T>
void Matrix<T>::operator*=(int const& x) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            mtr[i][j] *= x;
        }
    }
}

template <typename T>
Matrix<T> Matrix<T>::operator*(int const& x) const{
    Matrix<T> result(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result = mtr[i][j] * x;
        }
    }
    return result;
}

template <typename T>
void Matrix<T>::operator/=(int const& x) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            mtr[i][j] /= x;
        }
    }
}

template <typename T>
Matrix<T> Matrix<T>::operator/(int const& x) const{
    Matrix<T> result(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result = mtr[i][j] / x;
        }
    }
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix & x) {
    if (this == std::addressof(x)) { //проверка на самоприсваивание
        std::cerr << "self - assignment!\n";
        exit(1);
    }
    if (n != x.n || m != x.m) {
        std::cerr << "different sizes of matrices!\n";
        exit(1);
    }
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                mtr[i][j] = x.mtr[i][j];
            }
        }
        return *this;
    }
}


template <typename T>
Matrix<T> Matrix<T>::parallel_multiply(int x, int cnt_threads) const {
    Matrix<T> result(n, m);
    std::vector<std::thread> threads(cnt_threads);
    int cnt = n / cnt_threads;
    int mod = n % cnt_threads;
    int start = 0;
    for (int i = 0; i < cnt_threads; i++) {
        int end = start + cnt + (i == 0 ? mod : 0);
        threads[i] = std::thread([=, &result]()
                                 {
                                     for (int i = start; i < end; i++) {
                                         for (int j = 0; j < m; j++) {
                                             result.mtr[i][j] += mtr[i][j] * x;
                                         }
                                     }
                                 });
        start = end;
    }
    for (int i = 0; i < cnt_threads; i++){
        threads[i].join();
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::parallel_division(int x, int cnt_threads) const {
    Matrix<T> result(n, m);
    std::vector<std::thread> threads(cnt_threads);
    int cnt = n / cnt_threads;
    int mod = n % cnt_threads;
    int start = 0;
    for (int i = 0; i < cnt_threads; i++) {
        int end = start + cnt + (i == 0 ? mod : 0);
        threads[i] = std::thread([=, &result]()
                                 {
                                     for (int i = start; i < end; i++) {
                                         for (int j = 0; j < m; j++) {
                                             result.mtr[i][j] += mtr[i][j] / x;
                                         }
                                     }
                                 });
        start = end;
    }
    for (int i = 0; i < cnt_threads; i++){
        threads[i].join();
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::parallel_sum(const Matrix<T> & x, int cnt_threads) const {
    if (n != x.n || m != x.m) {
        std::cerr << "incorrect sizes of matrices!\n";
        exit(1);
    }
    else {
        Matrix<T> result(n, m);
        std::vector<std::thread> threads(cnt_threads);
        int cnt = n / cnt_threads;
        int mod = n % cnt_threads;
        int start = 0;
        for (int k = 0; k < cnt_threads; k++) {
            int end = start + cnt + (k == 0 ? mod : 0);
            threads[k] = std::thread([=, &result]()
                                     {
                                         for (int i = start; i < end; i++) {
                                             for (int j = 0; j < m; j++) {
                                                 result.mtr[i][j] = mtr[i][j] + x.mtr[i][j];
                                             }
                                         }
                                     });
            start = end;
        }
        for (int i = 0; i < cnt_threads; i++){
            threads[i].join();
        }
        return result;
    }
}

template <typename T>
Matrix<T> Matrix<T>::parallel_diff(const Matrix<T> & x, int cnt_threads) const {
    if (n != x.n || m != x.m) {
        std::cerr << "incorrect sizes of matrices!\n";
        exit(1);
    }
    else {
        Matrix<T> result(n, m);
        std::vector<std::thread> threads(cnt_threads);
        int cnt = n / cnt_threads;
        int mod = n % cnt_threads;
        int start = 0;
        int end = 0;
        for (int k = 0; k < cnt_threads; k++) {
            end = start + cnt + (k == 0 ? mod : 0);
            threads[k] = std::thread([=, &result]() {
                                         for (int i = start; i < end; i++) {
                                             for (int j = 0; j < m; j++) {
                                                 result.mtr[i][j] = mtr[i][j] - x.mtr[i][j];
                                             }
                                         }
                                     });
            start = end;
        }
        for (int i = 0; i < cnt_threads; i++){
            threads[i].join();
        }
        return result;
    }
}


template <typename T>
Matrix<T> Matrix<T>::parallel_mult(const Matrix<T> &x, int cnt_threads) const {
    Matrix<T> result(std::min(x.n, n) , std::min(x.m,m));
    if (m != x.n) {
        std::cerr << "matrices cannot be multiplied \n";
        exit(1);
    }
    else {
        std::vector<std::thread> threads(cnt_threads);
        int block_size = n / cnt_threads;
        int mod = n % cnt_threads;
        int start = 0;
        for (int i = 0; i < cnt_threads; i++) {
            int end = start + block_size + (i == 0 ? mod : 0);
            threads[i] = std::thread([=, &result]() {
                                         for (int i = start; i < end; ++i) {
                                             for (int j = 0; j < x.m; ++j) {
                                                 T sum = 0;
                                                 for (int k = 0; k < x.n; ++k) {
                                                     sum += mtr[i][k] * x.mtr[k][j];
                                                 }
                                                 result.mtr[i][j] = sum;
                                             }
                                         }
                                     });
            start = end;
        }
        for (int i = 0; i < cnt_threads; i++) {
            threads[i].join();
        }
        return result;
    }
}

template <typename T>
Matrix<T> Matrix<T>::async_sum(const Matrix<T> &b, int cnt) const {
    Matrix<T> result(n,m);
    int rows_in_block = n / cnt;
    std::vector<std::future<void>> futures(cnt);
    for (int i = 0; i < cnt; ++i) {
        futures[i] = std::async(std::launch::async, [this, &b , rows_in_block, cnt, &result, i]() {
            int start_row = i * rows_in_block;
            int end_row = (i == cnt - 1) ? n : (i + 1) * rows_in_block;
            for (int row = start_row; row < end_row; row++) {
                for (int col = 0; col < m; col++) {
                    result.mtr[row][col] = mtr[row][col] + b.mtr[row][col];
                }
            }
        });
    }
    for(auto& f : futures) {
        f.wait();
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::async_diff(const Matrix<T> &b, int cnt) const {
    Matrix<T> result(n,m);
    int rows_in_block = n / cnt;
    std::vector<std::future<void>> futures(cnt);
    for (int i = 0; i < cnt; ++i) {
        futures[i] = std::async(std::launch::async, [this, &b , rows_in_block, cnt, &result, i]() {
            int start_row = i * rows_in_block;
            int end_row = (i == cnt - 1) ? n : (i + 1) * rows_in_block;
            for (int row = start_row; row < end_row; row++) {
                for (int col = 0; col < m; col++) {
                    result.mtr[row][col] = mtr[row][col] - b.mtr[row][col];
                }
            }
        });
    }
    for(auto& f : futures) {
        f.wait();
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::async_multiply(int x, int cnt) const {
    Matrix<T> result(n,m);
    int rows_in_block = n / cnt;
    std::vector<std::future<void>> futures(cnt);
    for (int i = 0; i < cnt; ++i) {
        futures[i] = std::async(std::launch::async, [this, x , rows_in_block, cnt, &result, i]() {
            int start_row = i * rows_in_block;
            int end_row = (i == cnt - 1) ? n : (i + 1) * rows_in_block;
            for (int row = start_row; row < end_row; row++) {
                for (int col = 0; col < m; col++) {
                    result.mtr[row][col] = mtr[row][col] * x;
                }
            }
        });
    }
    for(auto& f : futures) {
        f.wait();
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::async_division(int x, int cnt) const {
    Matrix<T> result(n,m);
    int rows_in_block = n / cnt;
    std::vector<std::future<void>> futures(cnt);
    for (int i = 0; i < cnt; ++i) {
        futures[i] = std::async(std::launch::async, [this, x , rows_in_block, cnt, &result, i]() {
            int start_row = i * rows_in_block;
            int end_row = (i == cnt - 1) ? n : (i + 1) * rows_in_block;
            for (int row = start_row; row < end_row; row++) {
                for (int col = 0; col < m; col++) {
                    result.mtr[row][col] = mtr[row][col] / x;
                }
            }
        });
    }
    for(auto& f : futures) {
        f.wait();
    }
    return result;
}



#endif //MATRIX_MATRIX_H