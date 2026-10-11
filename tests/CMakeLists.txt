add_executable(glassbox_tests
  test_main.cpp
  test_foundations.cpp
  test_learning.cpp)
target_link_libraries(glassbox_tests PRIVATE glassbox glassbox_warnings)

add_test(NAME unit COMMAND glassbox_tests)

# End-to-end: drive the real CLI through generate -> train -> predict -> verify.
foreach(scenario IN ITEMS regression classification)
  add_test(NAME e2e_${scenario}
    COMMAND ${CMAKE_COMMAND}
      -DCLI=$<TARGET_FILE:glassbox_cli>
      -DSCENARIO=${scenario}
      -DWORK=${CMAKE_CURRENT_BINARY_DIR}/e2e_${scenario}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/e2e.cmake)
endforeach()
