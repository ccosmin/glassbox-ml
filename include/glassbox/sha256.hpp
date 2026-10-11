#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gbx {

/// SHA-256 (FIPS 180-4). Used to fingerprint datasets and model artifacts so
/// that every trained model can be traced to the exact bytes it came from.
class Sha256 {
 public:
  Sha256();
  void update(const void* data, std::size_t length);
  void update(const std::string& data) { update(data.data(), data.size()); }
  /// Finalises the hash and returns the digest as 64 lowercase hex digits.
  /// The object must not be updated afterwards.
  std::string hex_digest();

 private:
  void compress(const unsigned char* block);

  std::uint32_t state_[8];
  unsigned char buffer_[64];
  std::size_t buffer_length_ = 0;
  std::uint64_t total_length_ = 0;
};

std::string sha256_hex(const std::string& data);
std::string sha256_file(const std::string& path);

}  // namespace gbx
