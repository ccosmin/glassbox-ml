#pragma once

#include <string>
#include <vector>

#include "glassbox/pipeline.hpp"

namespace gbx {

struct Contribution {
  std::string feature;
  double raw_value = 0.0;
  double standardized_value = 0.0;  ///< how many training std-devs from the training mean
  double contribution = 0.0;        ///< weight * standardized value
};

/// Exact additive explanation of a single prediction.
///
/// For a linear model on standardised inputs,
///     decision = bias + sum_j w_j * z_j
/// so each term w_j * z_j is precisely how much feature j moved this
/// prediction away from the baseline (the prediction for an "average" input,
/// i.e. the bias). No approximation is involved -- this is what makes linear
/// models a "glass box". For classifiers the units are log-odds.
struct Explanation {
  double baseline = 0.0;  ///< bias: decision value at the training mean
  double decision = 0.0;  ///< baseline + sum of contributions
  double output = 0.0;    ///< prediction (regression) or probability (classification)
  std::vector<Contribution> contributions;  ///< sorted by |contribution|, descending
};

Explanation explain_prediction(const Pipeline& pipeline, const double* raw_x);

/// Global view: weights on standardised features are directly comparable
/// ("effect of +1 std-dev"), unlike weights on raw units.
struct FeatureWeight {
  std::string feature;
  double weight = 0.0;
};

std::vector<FeatureWeight> standardized_weights(const Pipeline& pipeline);

/// "feature:+0.123; other:-0.045" for the `top` largest contributions.
std::string summarize(const Explanation& explanation, std::size_t top);

}  // namespace gbx
