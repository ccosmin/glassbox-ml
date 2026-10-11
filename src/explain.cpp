#include "glassbox/explain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gbx {

Explanation explain_prediction(const Pipeline& pipeline, const double* raw_x) {
  const Vector& weights = pipeline.model.weights();
  const Vector& mean = pipeline.scaler.mean();
  const Vector& scale = pipeline.scaler.scale();

  Explanation e;
  e.baseline = pipeline.model.bias();
  e.decision = e.baseline;
  for (std::size_t j = 0; j < weights.size(); ++j) {
    Contribution c;
    c.feature = pipeline.feature_names[j];
    c.raw_value = raw_x[j];
    c.standardized_value = (raw_x[j] - mean[j]) / scale[j];
    c.contribution = weights[j] * c.standardized_value;
    e.decision += c.contribution;
    e.contributions.push_back(c);
  }
  e.output = pipeline.task() == Task::Regression ? e.decision : sigmoid(e.decision);
  std::stable_sort(e.contributions.begin(), e.contributions.end(),
                   [](const Contribution& a, const Contribution& b) {
                     return std::abs(a.contribution) > std::abs(b.contribution);
                   });
  return e;
}

std::vector<FeatureWeight> standardized_weights(const Pipeline& pipeline) {
  std::vector<FeatureWeight> out;
  for (std::size_t j = 0; j < pipeline.feature_names.size(); ++j) {
    out.push_back({pipeline.feature_names[j], pipeline.model.weights()[j]});
  }
  std::stable_sort(out.begin(), out.end(), [](const FeatureWeight& a, const FeatureWeight& b) {
    return std::abs(a.weight) > std::abs(b.weight);
  });
  return out;
}

std::string summarize(const Explanation& explanation, std::size_t top) {
  std::string out;
  const std::size_t n = std::min(top, explanation.contributions.size());
  for (std::size_t i = 0; i < n; ++i) {
    const Contribution& c = explanation.contributions[i];
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%+.3f", c.contribution);
    if (!out.empty()) out += "; ";
    out += c.feature + ":" + buffer;
  }
  return out;
}

}  // namespace gbx
