cmake_minimum_required(VERSION 3.16)

project(glassbox
  VERSION 0.1.0
  DESCRIPTION "Transparent, auditable linear models in modern C++"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GLASSBOX_BUILD_TESTS "Build the unit and end-to-end tests" ON)
option(GLASSBOX_WERROR "Treat compiler warnings as errors" OFF)
option(GLASSBOX_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# --- Build provenance -------------------------------------------------------
# The git commit is captured at configure time and embedded in every run
# manifest, so a trained model can be traced back to the exact source code.
set(GLASSBOX_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE _glassbox_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _glassbox_git_rc)
  if(_glassbox_git_rc EQUAL 0 AND _glassbox_commit)
    set(GLASSBOX_GIT_COMMIT ${_glassbox_commit})
  endif()
endif()
configure_file(cmake/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/glassbox/version.hpp @ONLY)

# --- Warnings and sanitizers ------------------------------------------------
add_library(glassbox_warnings INTERFACE)
if(MSVC)
  target_compile_options(glassbox_warnings INTERFACE
    /W4 /permissive- $<$<BOOL:${GLASSBOX_WERROR}>:/WX>)
else()
  target_compile_options(glassbox_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
    -Wcast-align -Woverloaded-virtual -Wnull-dereference
    $<$<BOOL:${GLASSBOX_WERROR}>:-Werror>)
endif()

if(GLASSBOX_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                      -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- Library ----------------------------------------------------------------
add_library(glassbox
  src/dataset.cpp
  src/evaluation.cpp
  src/explain.cpp
  src/json.cpp
  src/matrix.cpp
  src/metrics.cpp
  src/model.cpp
  src/pipeline.cpp
  src/random.cpp
  src/report.cpp
  src/run.cpp
  src/sha256.cpp
  src/synthetic.cpp)
target_include_directories(glassbox PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_features(glassbox PUBLIC cxx_std_17)
target_compile_definitions(glassbox PRIVATE GLASSBOX_BUILD_TYPE="$<CONFIG>")
target_link_libraries(glassbox PRIVATE glassbox_warnings)

# --- Command-line tool ------------------------------------------------------
add_executable(glassbox_cli apps/glassbox_cli.cpp)
set_target_properties(glassbox_cli PROPERTIES OUTPUT_NAME glassbox)
target_link_libraries(glassbox_cli PRIVATE glassbox glassbox_warnings)

# --- Tests ------------------------------------------------------------------
if(GLASSBOX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
