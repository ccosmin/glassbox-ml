#include "glassbox/model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "glassbox/format.hpp"

namespace gbx {

std::string to_string(Task task) {
  return task == Task::Regression ? "regression" : "classification";
}

Task task_from_string(const std::string& name) {
  if (name == "regression") return Task::Regression;
  if (name == "classification") return Task::Classification;
  throw std::invalid_argument("unknown task '" + name +
                              "' (expected 'regression' or 'classification')");
}

double sigmoid(double z) noexcept {
  if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);  // avoids overflow of exp(-z) for large negative z
  return e / (1.0 + e);
}

namespace {

// log(1 + exp(z)) without overflow.
double softplus(double z) noexcept { return std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z))); }

}  // namespace

LinearModel::LinearModel(Task task, std::size_t num_features)
    : task_(task), weights_(num_features, 0.0) {}

void LinearModel::set_parameters(Vector weights, double bias) {
  weights_ = std::move(weights);
  bias_ = bias;
}

double LinearModel::decision(const double* x) const noexcept {
  return dot(weights_.data(), x, weights_.size()) + bias_;
}

double LinearModel::predict_one(const double* x) const noexcept {
  const double z = decision(x);
  return task_ == Task::Regression ? z : sigmoid(z);
}

Vector LinearModel::predict(const Matrix& X) const {
  if (X.cols() != weights_.size()) {
    throw std::invalid_argument("model expects " + std::to_string(weights_.size()) +
                                " features, got " + std::to_string(X.cols()));
  }
  Vector out(X.rows());
  for (std::size_t i = 0; i < X.rows(); ++i) out[i] = predict_one(X.row(i));
  return out;
}

void LinearModel::check_inputs(const Matrix& X, const Vector& y) const {
  if (X.rows() == 0) throw std::invalid_argument("no training rows");
  if (X.rows() != y.size()) throw std::invalid_argument("X and y have different row counts");
  if (X.cols() != weights_.size()) {
    throw std::invalid_argument("model expects " + std::to_string(weights_.size()) +
                                " features, got " + std::to_string(X.cols()));
  }
  if (task_ == Task::Classification) {
    for (const double v : y) {
      if (v != 0.0 && v != 1.0) {
        throw std::invalid_argument("classification targets must be 0 or 1, found " +
                                    format_double(v));
      }
    }
  }
}

double LinearModel::cost(const Matrix& X, const Vector& y, double l2) const {
  check_inputs(X, y);
  return cost_unchecked(X, y, l2);
}

double LinearModel::cost_unchecked(const Matrix& X, const Vector& y, double l2) const {
  const auto m = static_cast<double>(X.rows());
  double total = 0.0;
  for (std::size_t i = 0; i < X.rows(); ++i) {
    const double z = decision(X.row(i));
    if (task_ == Task::Regression) {
      const double residual = z - y[i];
      total += residual * residual;
    } else {
      // -[y log sigmoid(z) + (1 - y) log(1 - sigmoid(z))] == softplus(z) - y z
      total += softplus(z) - y[i] * z;
    }
  }
  const double data_term = task_ == Task::Regression ? total / (2.0 * m) : total / m;
  const double penalty = l2 / (2.0 * m) * dot(weights_.data(), weights_.data(), weights_.size());
  return data_term + penalty;
}

void LinearModel::gradient(const Matrix& X, const Vector& y, double l2, Vector& grad_w,
                           double& grad_b) const {
  check_inputs(X, y);
  gradient_unchecked(X, y, l2, grad_w, grad_b);
}

void LinearModel::gradient_unchecked(const Matrix& X, const Vector& y, double l2, Vector& grad_w,
                                     double& grad_b) const {
  const std::size_t n = X.cols();
  grad_w.assign(n, 0.0);
  grad_b = 0.0;
  for (std::size_t i = 0; i < X.rows(); ++i) {
    const double* x = X.row(i);
    const double error = predict_one(x) - y[i];
    for (std::size_t j = 0; j < n; ++j) grad_w[j] += error * x[j];
    grad_b += error;
  }
  const auto m = static_cast<double>(X.rows());
  for (std::size_t j = 0; j < n; ++j) grad_w[j] = grad_w[j] / m + (l2 / m) * weights_[j];
  grad_b /= m;
}

TrainHistory LinearModel::fit(const Matrix& X, const Vector& y, const TrainConfig& config) {
  check_inputs(X, y);
  if (!(config.learning_rate > 0.0)) throw std::invalid_argument("learning rate must be positive");
  if (!(config.l2 >= 0.0)) throw std::invalid_argument("l2 must be non-negative");
  if (config.max_epochs == 0) throw std::invalid_argument("max epochs must be positive");
  if (!(config.tolerance >= 0.0)) throw std::invalid_argument("tolerance must be non-negative");

  weights_.assign(X.cols(), 0.0);
  bias_ = 0.0;

  TrainHistory history;
  history.cost.reserve(config.max_epochs + 1);
  double previous = cost_unchecked(X, y, config.l2);
  history.cost.push_back(previous);

  Vector grad_w;
  double grad_b = 0.0;
  for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
    gradient_unchecked(X, y, config.l2, grad_w, grad_b);
    for (std::size_t j = 0; j < weights_.size(); ++j) weights_[j] -= config.learning_rate * grad_w[j];
    bias_ -= config.learning_rate * grad_b;

    const double current = cost_unchecked(X, y, config.l2);
    history.cost.push_back(current);
    if (!std::isfinite(current)) {
      throw std::runtime_error("gradient descent diverged at epoch " + std::to_string(epoch) +
                               " (cost is not finite); try a smaller learning rate");
    }
    if (current > previous) history.cost_increased = true;
    if (std::abs(previous - current) <= config.tolerance * std::max(1.0, std::abs(previous))) {
      history.converged = true;
      break;
    }
    previous = current;
  }
  return history;
}

void LinearModel::fit_closed_form(const Matrix& X, const Vector& y, double l2) {
  if (task_ != Task::Regression) {
    throw std::logic_error("the closed-form solver only applies to regression");
  }
  check_inputs(X, y);
  if (!(l2 >= 0.0)) throw std::invalid_argument("l2 must be non-negative");

  // Augment x with a constant 1 for the bias (last index) and solve
  // (A^T A + lambda D) theta = A^T y, where D excludes the bias term.
  const std::size_t n = X.cols();
  const std::size_t k = n + 1;
  Matrix gram(k, k);
  Vector rhs(k, 0.0);
  for (std::size_t i = 0; i < X.rows(); ++i) {
    const double* x = X.row(i);
    for (std::size_t p = 0; p < k; ++p) {
      const double xp = p < n ? x[p] : 1.0;
      rhs[p] += xp * y[i];
      for (std::size_t q = 0; q <= p; ++q) gram(p, q) += xp * (q < n ? x[q] : 1.0);
    }
  }
  for (std::size_t p = 0; p < k; ++p) {
    for (std::size_t q = 0; q < p; ++q) gram(q, p) = gram(p, q);
  }
  for (std::size_t j = 0; j < n; ++j) gram(j, j) += l2;

  Vector theta;
  try {
    theta = solve_spd(std::move(gram), std::move(rhs));
  } catch (const std::runtime_error&) {
    throw std::runtime_error(
        "normal equations are singular (are some features perfectly collinear?); "
        "use l2 > 0 or the gradient-descent solver");
  }
  bias_ = theta.back();
  theta.pop_back();
  weights_ = std::move(theta);
}

}  // namespace gbx
