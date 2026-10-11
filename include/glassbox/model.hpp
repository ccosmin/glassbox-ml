#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "glassbox/matrix.hpp"

namespace gbx {

enum class Task { Regression, Classification };

std::string to_string(Task task);
Task task_from_string(const std::string& name);

struct TrainConfig {
  double learning_rate = 0.1;
  std::size_t max_epochs = 5000;
  double l2 = 0.0;           ///< L2 regularisation strength (lambda); the bias is not penalised
  double tolerance = 1e-10;  ///< stop when the relative cost change falls below this
};

struct TrainHistory {
  std::vector<double> cost;     ///< cost before training, then after every epoch
  bool converged = false;
  bool cost_increased = false;  ///< a symptom of a learning rate that is too high

  std::size_t epochs_run() const noexcept { return cost.empty() ? 0 : cost.size() - 1; }
};

/// Numerically stable logistic function.
double sigmoid(double z) noexcept;

/// A linear model f(x) = g(w . x + b) trained by batch gradient descent.
///
///  * Regression:     g(z) = z,          cost = 1/(2m) sum (f(x) - y)^2
///  * Classification: g(z) = sigmoid(z), cost = -1/m sum [y log f + (1-y) log(1-f)]
///
/// Both add the L2 penalty lambda/(2m) ||w||^2. With these choices the
/// gradient has the same form for both tasks:
///     dJ/dw = 1/m sum (f(x) - y) x + (lambda/m) w,   dJ/db = 1/m sum (f(x) - y)
/// which is why a single class implements linear and logistic regression.
class LinearModel {
 public:
  LinearModel() = default;
  LinearModel(Task task, std::size_t num_features);

  Task task() const noexcept { return task_; }
  std::size_t num_features() const noexcept { return weights_.size(); }
  const Vector& weights() const noexcept { return weights_; }
  double bias() const noexcept { return bias_; }
  void set_parameters(Vector weights, double bias);

  /// Linear score z = w . x + b (log-odds for classification).
  double decision(const double* x) const noexcept;
  /// Predicted value (regression) or probability P(y = 1 | x) (classification).
  double predict_one(const double* x) const noexcept;
  Vector predict(const Matrix& X) const;

  double cost(const Matrix& X, const Vector& y, double l2) const;
  void gradient(const Matrix& X, const Vector& y, double l2, Vector& grad_w, double& grad_b) const;

  /// Batch gradient descent from zero-initialised parameters. The cost is
  /// convex for both tasks, so the result does not depend on initialisation.
  TrainHistory fit(const Matrix& X, const Vector& y, const TrainConfig& config);

  /// Exact ridge solution via the normal equations (regression only). Used to
  /// cross-check gradient descent.
  void fit_closed_form(const Matrix& X, const Vector& y, double l2);

 private:
  void check_inputs(const Matrix& X, const Vector& y) const;
  double cost_unchecked(const Matrix& X, const Vector& y, double l2) const;
  void gradient_unchecked(const Matrix& X, const Vector& y, double l2, Vector& grad_w,
                          double& grad_b) const;

  Task task_ = Task::Regression;
  Vector weights_;
  double bias_ = 0.0;
};

}  // namespace gbx
