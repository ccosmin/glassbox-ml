#include "glassbox/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "glassbox/metrics.hpp"

namespace gbx {

namespace {

struct Summary {
  double mean = 0.0;
  double std = 0.0;
};

Summary summarize(const Vector& values) {
  Summary s;
  const auto n = static_cast<double>(values.size());
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
  double var = 0.0;
  for (const double v : values) var += (v - s.mean) * (v - s.mean);
  s.std = values.size() > 1 ? std::sqrt(var / (n - 1.0)) : 0.0;
  return s;
}

void check_args(const Pipeline& pipeline, const Dataset& data, std::size_t repeats) {
  if (repeats == 0) throw std::invalid_argument("repeats must be positive");
  if (data.size() == 0) throw std::invalid_argument("evaluation data is empty");
  if (data.num_features() != pipeline.feature_names.size()) {
    throw std::invalid_argument("evaluation data has the wrong number of features");
  }
}

}  // namespace

std::vector<FeatureImportance> permutation_importance(const Pipeline& pipeline,
                                                      const Dataset& data, std::size_t repeats,
                                                      Rng& rng) {
  check_args(pipeline, data, repeats);
  const Task task = pipeline.task();
  const double baseline = primary_score(task, data.y, pipeline.predict(data.X));

  std::vector<FeatureImportance> result;
  std::vector<std::size_t> permutation(data.size());
  for (std::size_t j = 0; j < data.num_features(); ++j) {
    Vector drops;
    for (std::size_t r = 0; r < repeats; ++r) {
      std::iota(permutation.begin(), permutation.end(), std::size_t{0});
      rng.shuffle(permutation);
      Matrix shuffled = data.X;
      for (std::size_t i = 0; i < data.size(); ++i) shuffled(i, j) = data.X(permutation[i], j);
      drops.push_back(baseline - primary_score(task, data.y, pipeline.predict(shuffled)));
    }
    const Summary s = summarize(drops);
    result.push_back({pipeline.feature_names[j], s.mean, s.std});
  }
  std::stable_sort(result.begin(), result.end(),
                   [](const FeatureImportance& a, const FeatureImportance& b) {
                     return a.mean_drop > b.mean_drop;
                   });
  return result;
}

std::vector<NoisePoint> noise_sensitivity(const Pipeline& pipeline, const Dataset& data,
                                          const std::vector<double>& noise_levels,
                                          std::size_t repeats, Rng& rng) {
  check_args(pipeline, data, repeats);
  const Task task = pipeline.task();
  const Vector& sigma = pipeline.scaler.scale();

  std::vector<NoisePoint> result;
  for (const double level : noise_levels) {
    if (!(level >= 0.0)) throw std::invalid_argument("noise levels must be non-negative");
    if (level == 0.0) {
      result.push_back({0.0, primary_score(task, data.y, pipeline.predict(data.X)), 0.0});
      continue;
    }
    Vector scores;
    for (std::size_t r = 0; r < repeats; ++r) {
      Matrix noisy = data.X;
      for (std::size_t i = 0; i < noisy.rows(); ++i) {
        for (std::size_t j = 0; j < noisy.cols(); ++j) noisy(i, j) += rng.normal(0.0, level * sigma[j]);
      }
      scores.push_back(primary_score(task, data.y, pipeline.predict(noisy)));
    }
    const Summary s = summarize(scores);
    result.push_back({level, s.mean, s.std});
  }
  return result;
}

}  // namespace gbx
