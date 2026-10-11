#pragma once

// A deliberately tiny test harness so the project builds with zero external
// dependencies (no network access needed in CI or air-gapped environments).

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace gbx_test {

struct TestCase {
  const char* name;
  void (*function)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, void (*function)()) { registry().push_back({name, function}); }
};

struct Failure {
  std::string message;
};

}  // namespace gbx_test

#define GBX_CONCAT_INNER(a, b) a##b
#define GBX_CONCAT(a, b) GBX_CONCAT_INNER(a, b)

#define TEST_CASE(name)                                                                     \
  static void GBX_CONCAT(gbx_test_function_, __LINE__)();                                  \
  static const gbx_test::Registrar GBX_CONCAT(gbx_test_registrar_, __LINE__)(               \
      name, &GBX_CONCAT(gbx_test_function_, __LINE__));                                     \
  static void GBX_CONCAT(gbx_test_function_, __LINE__)()

#define GBX_FAIL(text)                                                    \
  do {                                                                    \
    std::ostringstream gbx_message_;                                      \
    gbx_message_ << __FILE__ << ":" << __LINE__ << ": " << text;          \
    throw gbx_test::Failure{gbx_message_.str()};                          \
  } while (false)

#define CHECK(condition)                                     \
  do {                                                       \
    if (!(condition)) GBX_FAIL("CHECK(" #condition ") failed"); \
  } while (false)

#define CHECK_EQ(a, b)                                                                     \
  do {                                                                                     \
    /* copies, not references: (a) may refer into a temporary */                           \
    const auto gbx_a_ = (a);                                                               \
    const auto gbx_b_ = (b);                                                               \
    if (!(gbx_a_ == gbx_b_)) GBX_FAIL("CHECK_EQ(" #a ", " #b ") failed: " << gbx_a_ << " != " << gbx_b_); \
  } while (false)

#define CHECK_NEAR(a, b, tolerance)                                                        \
  do {                                                                                     \
    const double gbx_a_ = (a);                                                             \
    const double gbx_b_ = (b);                                                             \
    if (!(std::abs(gbx_a_ - gbx_b_) <= (tolerance)))                                       \
      GBX_FAIL("CHECK_NEAR(" #a ", " #b ") failed: " << gbx_a_ << " vs " << gbx_b_         \
                                                     << " (tolerance " << (tolerance) << ")"); \
  } while (false)

#define CHECK_THROWS(expression)                                          \
  do {                                                                    \
    bool gbx_threw_ = false;                                              \
    try {                                                                 \
      (void)(expression);                                                 \
    } catch (const std::exception&) {                                     \
      gbx_threw_ = true;                                                  \
    }                                                                     \
    if (!gbx_threw_) GBX_FAIL("CHECK_THROWS(" #expression ") did not throw"); \
  } while (false)
