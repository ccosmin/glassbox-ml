// Tests for the building blocks: JSON, SHA-256, random numbers, linear algebra.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "glassbox/format.hpp"
#include "glassbox/json.hpp"
#include "glassbox/matrix.hpp"
#include "glassbox/random.hpp"
#include "glassbox/sha256.hpp"
#include "test_framework.hpp"

using namespace gbx;

TEST_CASE("json: round trip preserves structure and order") {
  json::Value v = json::Value::object();
  v.set("name", "glassbox");
  v.set("version", 1);
  v.set("enabled", true);
  v.set("nothing", nullptr);
  v.set("weights", json::from_vector({1.5, -2.25, 1e-12}));
  json::Value nested = json::Value::object();
  nested.set("z", "last");
  nested.set("a", "first");
  v.set("nested", nested);

  const json::Value parsed = json::Value::parse(v.dump(2));
  CHECK_EQ(parsed.dump(0), v.dump(0));
  CHECK_EQ(parsed.at("nested").as_object().front().first, std::string("z"));
  CHECK_EQ(parsed.at("weights").as_array().size(), std::size_t{3});
}

TEST_CASE("json: doubles survive serialisation exactly") {
  const std::vector<double> values = {0.1, 1.0 / 3.0, -123456.789e-7, 6.02214076e23,
                                      std::numeric_limits<double>::min(), 0.0};
  const json::Value parsed = json::Value::parse(json::from_vector(values).dump());
  const std::vector<double> back = json::to_vector(parsed);
  for (std::size_t i = 0; i < values.size(); ++i) CHECK(back[i] == values[i]);
  CHECK_EQ(format_double(0.1), std::string("0.1"));
}

TEST_CASE("json: string escapes and unicode") {
  const std::string text = "quote\" backslash\\ newline\n tab\t";
  CHECK_EQ(json::Value::parse(json::Value(text).dump()).as_string(), text);
  CHECK_EQ(json::Value::parse("\"\\u00e9\\ud83d\\ude00\"").as_string(), std::string("\xC3\xA9\xF0\x9F\x98\x80"));
}

TEST_CASE("json: non-finite numbers become null and read back as NaN") {
  json::Value v = json::Value::object();
  v.set("x", std::nan(""));
  CHECK_EQ(v.dump(0), std::string("{\"x\":null}"));
  CHECK(std::isnan(json::Value::parse(v.dump()).at("x").as_number()));
}

TEST_CASE("json: malformed input is rejected") {
  CHECK_THROWS(json::Value::parse("{"));
  CHECK_THROWS(json::Value::parse("[1, 2,]"));
  CHECK_THROWS(json::Value::parse("{\"a\" 1}"));
  CHECK_THROWS(json::Value::parse("\"unterminated"));
  CHECK_THROWS(json::Value::parse("1 2"));
  CHECK_THROWS(json::Value::parse("nul"));
  CHECK_THROWS(json::Value::parse("[1.2.3]"));
  CHECK_THROWS(json::Value::parse("{}").at("missing"));
  CHECK_THROWS(json::Value::parse("\"text\"").as_number());
}

TEST_CASE("sha256: FIPS 180-4 test vectors") {
  CHECK_EQ(sha256_hex(""), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK_EQ(sha256_hex("abc"), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  CHECK_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
           std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
  CHECK_EQ(sha256_hex(std::string(1000000, 'a')),
           std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
}

TEST_CASE("sha256: incremental updates match one-shot hashing") {
  const std::string text = "The quick brown fox jumps over the lazy dog, repeatedly, across blocks.";
  for (std::size_t cut = 0; cut <= text.size(); cut += 7) {
    Sha256 h;
    h.update(text.substr(0, cut));
    h.update(text.substr(cut));
    CHECK_EQ(h.hex_digest(), sha256_hex(text));
  }
}

TEST_CASE("random: engine conforms to the C++ standard") {
  // [rand.predef]: the 10000th output of a default-constructed mt19937_64.
  Rng rng(5489u);
  std::uint64_t value = 0;
  for (int i = 0; i < 10000; ++i) value = rng.next_u64();
  CHECK_EQ(value, std::uint64_t{9981545732273789042ULL});
}

TEST_CASE("random: same seed gives the same sequence") {
  Rng a(123), b(123), c(124);
  bool differs = false;
  for (int i = 0; i < 100; ++i) {
    const double x = a.normal();
    CHECK(x == b.normal());
    differs = differs || x != c.normal();
  }
  CHECK(differs);
}

TEST_CASE("random: distributions have the expected range and moments") {
  Rng rng(42);
  const int n = 200000;
  double sum = 0.0, sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double u = rng.uniform();
    CHECK(u >= 0.0 && u < 1.0);
    const double z = rng.normal(2.0, 3.0);
    sum += z;
    sum_sq += z * z;
  }
  const double mean = sum / n;
  CHECK_NEAR(mean, 2.0, 0.05);
  CHECK_NEAR(std::sqrt(sum_sq / n - mean * mean), 3.0, 0.05);
  for (int i = 0; i < 1000; ++i) CHECK(rng.index(7) < 7);
  CHECK_THROWS(rng.index(0));
}

TEST_CASE("random: shuffle produces a permutation") {
  Rng rng(9);
  std::vector<int> v(50);
  std::iota(v.begin(), v.end(), 0);
  const std::vector<int> original = v;
  rng.shuffle(v);
  CHECK(v != original);
  std::sort(v.begin(), v.end());
  CHECK(v == original);
}

TEST_CASE("linalg: Cholesky solve") {
  Matrix a(3, 3);
  const double values[3][3] = {{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = values[i][j];
  // b = A * [1, 2, 3]
  const Vector x = solve_spd(a, {4 + 24 - 48, 12 + 74 - 129, -16 - 86 + 294});
  CHECK_NEAR(x[0], 1.0, 1e-9);
  CHECK_NEAR(x[1], 2.0, 1e-9);
  CHECK_NEAR(x[2], 3.0, 1e-9);

  Matrix singular(2, 2, 1.0);
  CHECK_THROWS(solve_spd(singular, {1.0, 1.0}));
}
