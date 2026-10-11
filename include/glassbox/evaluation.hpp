#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "glassbox/dataset.hpp"
#include "glassbox/pipeline.hpp"
#include "glassbox/random.hpp"

namespace gbx {

/// Model-agnostic feature importance: how much the primary score drops when
/// one feature's values are shuffled across rows (breaking its link to the
/// target while keeping its distribution). Features the model does not rely
/// on score ~0.
struct FeatureImportance {
  std::string feature;
  double mean_drop = 0.0;
  double std_drop = 0.0;
};

std::vector<FeatureImportance> permutation_importance(const Pipeline& pipeline,
                                                      const Dataset& data, std::size_t repeats,
                                                      Rng& rng);

/// Robustness probe: score under additive Gaussian noise. A noise level of
/// 0.5 adds N(0, (0.5 * sigma_j)^2) to feature j, where sigma_j is that
/// feature's training standard deviation, so levels are comparable across
/// features with different units.
struct NoisePoint {
  double noise_level = 0.0;
  double mean_score = 0.0;
  double std_score = 0.0;
};

std::vector<NoisePoint> noise_sensitivity(const Pipeline& pipeline, const Dataset& data,
                                          const std::vector<double>& noise_levels,
                                          std::size_t repeats, Rng& rng);

}  // namespace gbx
