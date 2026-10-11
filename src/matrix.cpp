#include "glassbox/matrix.hpp"

#include <cmath>

namespace gbx {

Vector solve_spd(Matrix a, Vector b) {
  const std::size_t n = a.rows();
  if (a.cols() != n || b.size() != n) {
    throw std::invalid_argument("solve_spd: dimension mismatch");
  }

  // In-place Cholesky: the lower triangle of `a` becomes L with A = L L^T.
  for (std::size_t j = 0; j < n; ++j) {
    double diag = a(j, j);
    for (std::size_t k = 0; k < j; ++k) diag -= a(j, k) * a(j, k);
    if (!(diag > 0.0)) {
      throw std::runtime_error("solve_spd: matrix is not positive definite");
    }
    a(j, j) = std::sqrt(diag);
    for (std::size_t i = j + 1; i < n; ++i) {
      double value = a(i, j);
      for (std::size_t k = 0; k < j; ++k) value -= a(i, k) * a(j, k);
      a(i, j) = value / a(j, j);
    }
  }

  // Forward substitution: L y = b.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < i; ++k) b[i] -= a(i, k) * b[k];
    b[i] /= a(i, i);
  }
  // Back substitution: L^T x = y.
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t k = i + 1; k < n; ++k) b[i] -= a(k, i) * b[k];
    b[i] /= a(i, i);
  }
  return b;
}

}  // namespace gbx
