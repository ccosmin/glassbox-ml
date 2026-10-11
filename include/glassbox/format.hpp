#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace gbx {

/// Shortest decimal text (15-17 significant digits) that parses back to
/// exactly the same double. Used for every number we persist, so that saving
/// and reloading a model or a run specification is lossless.
inline std::string format_double(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
  char buffer[32];
  for (int precision = 15; precision <= 17; ++precision) {
    std::snprintf(buffer, sizeof buffer, "%.*g", precision, value);
    if (std::strtod(buffer, nullptr) == value) break;
  }
  return buffer;
}

/// Fixed-point text for human-readable reports.
inline std::string format_fixed(double value, int decimals = 4) {
  if (!std::isfinite(value)) return "n/a";
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, "%.*f", decimals, value);
  return buffer;
}

}  // namespace gbx
