#pragma once

#include <cstddef>

#include "glassbox/dataset.hpp"
#include "glassbox/random.hpp"

namespace gbx {

/// Synthetic datasets with a *known* ground truth, so that what the model
/// learns can be checked against how the data was generated. Both include a
/// `noise_feature` that has no effect on the target: a trustworthy pipeline
/// should assign it near-zero importance.

/// House prices (kEUR) from area, bedrooms, age and distance to the centre.
/// price = 50 + 2.8 area + 12 bedrooms - 1.1 age - 4.5 distance + N(0, 20^2)
Dataset make_house_prices(std::size_t rows, Rng& rng);

/// Industrial machine faults (0/1) from sensor readings.
/// logit = -1 + 0.25 (temp - 70) + 1.2 (vibration - 3)
///         - 0.6 (pressure - 6) + 0.08 (hours - 20)
Dataset make_machine_faults(std::size_t rows, Rng& rng);

}  // namespace gbx
