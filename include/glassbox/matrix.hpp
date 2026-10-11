#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gbx {

using Vector = std::vector<double>;

/// Dense, row-major matrix of doubles.
///
/// Deliberately minimal: the models in this library only need row access and
/// dot products, so there is no expression-template or BLAS machinery to hide
/// what the algorithms actually do.
class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }

  double& operator()(std::size_t r, std::size_t c) noexcept { return data_[r * cols_ + c]; }
  double operator()(std::size_t r, std::size_t c) const noexcept { return data_[r * cols_ + c]; }

  double* row(std::size_t r) noexcept { return data_.data() + r * cols_; }
  const double* row(std::size_t r) const noexcept { return data_.data() + r * cols_; }

  /// Appends a row. On a default-constructed matrix the first row fixes the
  /// column count.
  void append_row(const Vector& values) {
    if (rows_ == 0 && cols_ == 0) cols_ = values.size();
    if (values.size() != cols_) {
      throw std::invalid_argument("Matrix::append_row: column count mismatch");
    }
    data_.insert(data_.end(), values.begin(), values.end());
    ++rows_;
  }

  Vector column(std::size_t c) const {
    Vector out(rows_);
    for (std::size_t r = 0; r < rows_; ++r) out[r] = (*this)(r, c);
    return out;
  }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

inline double dot(const double* a, const double* b, std::size_t n) noexcept {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

/// Solves A x = b for a symmetric positive-definite A using a Cholesky
/// factorisation. Throws std::runtime_error if A is not positive definite.
Vector solve_spd(Matrix a, Vector b);

}  // namespace gbx
