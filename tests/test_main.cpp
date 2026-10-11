#include <cstring>
#include <exception>
#include <iostream>

#include "test_framework.hpp"

// Usage: glassbox_tests [substring]   -- runs tests whose name contains substring.
int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0;
  int failed = 0;
  for (const auto& test : gbx_test::registry()) {
    if (filter != nullptr && std::strstr(test.name, filter) == nullptr) continue;
    try {
      test.function();
      ++passed;
    } catch (const gbx_test::Failure& f) {
      ++failed;
      std::cerr << "FAIL " << test.name << "\n  " << f.message << "\n";
    } catch (const std::exception& e) {
      ++failed;
      std::cerr << "FAIL " << test.name << "\n  unexpected exception: " << e.what() << "\n";
    }
  }
  std::cout << passed << " passed, " << failed << " failed\n";
  return failed == 0 && passed > 0 ? 0 : 1;
}
