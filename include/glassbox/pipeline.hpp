#pragma once

#include <string>
#include <vector>

#include "glassbox/dataset.hpp"
#include "glassbox/json.hpp"
#include "glassbox/model.hpp"

namespace gbx {

/// Everything needed to score raw data: the feature contract, the scaling
/// fitted on training data, and the model. This is the deployable unit and
/// is what gets saved as model.json.
struct Pipeline {
  std::vector<std::string> feature_names;
  std::string target_name;
  Standardizer scaler;
  LinearModel model;
  double threshold = 0.5;  ///< decision threshold (classification only)

  Task task() const noexcept { return model.task(); }

  /// Prediction for raw (unscaled) features.
  double predict_one(const double* raw_x) const;
  Vector predict(const Matrix& raw_X) const;
  /// Hard 0/1 labels from probabilities (classification only).
  Vector labels(const Vector& probabilities) const;

  json::Value to_json() const;
  static Pipeline from_json(const json::Value& value);

  /// Canonical text form; its SHA-256 identifies the model.
  std::string serialize() const { return to_json().dump(2) + "\n"; }
  static Pipeline load(const std::string& path);
};

}  // namespace gbx
