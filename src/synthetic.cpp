#include "glassbox/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "glassbox/model.hpp"

namespace gbx {

namespace {

double round2(double x) { return std::round(x * 100.0) / 100.0; }

void check_rows(std::size_t rows) {
  if (rows < 10) throw std::invalid_argument("need at least 10 rows");
}

}  // namespace

Dataset make_house_prices(std::size_t rows, Rng& rng) {
  check_rows(rows);
  Dataset d;
  d.feature_names = {"area_m2", "bedrooms", "age_years", "distance_km", "noise_feature"};
  d.target_name = "price_keur";
  d.X = Matrix(rows, d.feature_names.size());
  d.y.resize(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    const double area = round2(rng.uniform(40.0, 200.0));
    // Bedrooms correlate with area, as in real housing data.
    const double bedrooms = std::max(1.0, std::round(area / 35.0 + rng.normal(0.0, 0.7)));
    const double age = round2(rng.uniform(0.0, 60.0));
    const double distance = round2(rng.uniform(0.5, 25.0));
    const double noise = round2(rng.normal());
    const double price =
        50.0 + 2.8 * area + 12.0 * bedrooms - 1.1 * age - 4.5 * distance + rng.normal(0.0, 20.0);
    const double row[] = {area, bedrooms, age, distance, noise};
    std::copy(std::begin(row), std::end(row), d.X.row(i));
    d.y[i] = round2(price);
  }
  return d;
}

Dataset make_machine_faults(std::size_t rows, Rng& rng) {
  check_rows(rows);
  Dataset d;
  d.feature_names = {"temperature_c", "vibration_mm_s", "pressure_bar", "operating_hours_k",
                     "noise_feature"};
  d.target_name = "fault";
  d.X = Matrix(rows, d.feature_names.size());
  d.y.resize(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    const double temperature = round2(rng.normal(70.0, 8.0));
    const double vibration = round2(std::abs(rng.normal(3.0, 1.2)));
    const double pressure = round2(rng.normal(6.0, 0.8));
    const double hours = round2(rng.uniform(0.0, 40.0));
    const double noise = round2(rng.normal());
    const double logit = -1.0 + 0.25 * (temperature - 70.0) + 1.2 * (vibration - 3.0) -
                         0.6 * (pressure - 6.0) + 0.08 * (hours - 20.0);
    const double row[] = {temperature, vibration, pressure, hours, noise};
    std::copy(std::begin(row), std::end(row), d.X.row(i));
    d.y[i] = rng.uniform() < sigmoid(logit) ? 1.0 : 0.0;
  }
  return d;
}

}  // namespace gbx
