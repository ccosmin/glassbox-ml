#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace gbx {

/// Seeded random number generator with portable derived distributions.
///
/// The raw std::mt19937_64 sequence is fully specified by the C++ standard,
/// but std::uniform_real_distribution, std::normal_distribution and
/// std::shuffle are not: libstdc++, libc++ and MSVC produce different values
/// for the same seed. To keep data splits and permutations identical across
/// toolchains, every distribution used by glassbox is implemented here on top
/// of the raw engine output.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_(seed) {}

  std::uint64_t next_u64() { return engine_(); }

  /// Uniform double in [0, 1) with 53 bits of randomness.
  double uniform() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }

  /// Uniform double in [lo, hi).
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }

  /// Normally distributed double (Box-Muller transform).
  double normal(double mean = 0.0, double stddev = 1.0);

  /// Unbiased uniform integer in [0, n) using rejection sampling.
  std::size_t index(std::size_t n);

  /// Fisher-Yates shuffle.
  template <typename T>
  void shuffle(std::vector<T>& values) {
    for (std::size_t i = values.size(); i > 1; --i) {
      std::swap(values[i - 1], values[index(i)]);
    }
  }

 private:
  std::mt19937_64 engine_;
  bool has_spare_ = false;
  double spare_ = 0.0;
};

}  // namespace gbx
