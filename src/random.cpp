#include "glassbox/random.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace gbx {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

double Rng::normal(double mean, double stddev) {
  if (has_spare_) {
    has_spare_ = false;
    return mean + stddev * spare_;
  }
  double u1 = 0.0;
  do {
    u1 = uniform();
  } while (u1 <= 0.0);  // log(0) is undefined
  const double u2 = uniform();
  const double radius = std::sqrt(-2.0 * std::log(u1));
  const double theta = 2.0 * kPi * u2;
  spare_ = radius * std::sin(theta);
  has_spare_ = true;
  return mean + stddev * radius * std::cos(theta);
}

std::size_t Rng::index(std::size_t n) {
  if (n == 0) throw std::invalid_argument("Rng::index: n must be positive");
  const auto bound = static_cast<std::uint64_t>(n);
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  // Reject the incomplete final block so that every residue is equally likely.
  const std::uint64_t limit = kMax - (kMax % bound);
  std::uint64_t x = 0;
  do {
    x = engine_();
  } while (x >= limit);
  return static_cast<std::size_t>(x % bound);
}

}  // namespace gbx
