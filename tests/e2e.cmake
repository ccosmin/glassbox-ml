# End-to-end test of the glassbox CLI.
# Usage: cmake -DCLI=<glassbox> -DSCENARIO=regression|classification -DWORK=<dir> -P e2e.cmake

if(SCENARIO STREQUAL "regression")
  set(KIND houses)
  set(TARGET price_keur)
  set(EXTRA --solver closed-form --l2 0.1)
else()
  set(KIND faults)
  set(TARGET fault)
  set(EXTRA --learning-rate 0.5 --l2 1)
endif()

file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

function(run_cli expected_rc)
  execute_process(COMMAND "${CLI}" ${ARGN}
    WORKING_DIRECTORY "${WORK}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL expected_rc)
    message(FATAL_ERROR "glassbox ${ARGN}\nexit code ${rc}, expected ${expected_rc}\n${out}\n${err}")
  endif()
  set(LAST_OUTPUT "${out}" PARENT_SCOPE)
endfunction()

run_cli(0 generate --kind ${KIND} --rows 600 --seed 3 --out data.csv)
run_cli(0 train --task ${SCENARIO} --data data.csv --target ${TARGET} ${EXTRA} --repeats 2 --out run)

foreach(artifact model.json manifest.json model_card.md cost_history.csv)
  if(NOT EXISTS "${WORK}/run/${artifact}")
    message(FATAL_ERROR "missing artifact ${artifact}")
  endif()
endforeach()

run_cli(0 predict --model run/model.json --data data.csv --explain --limit 5)
string(REGEX MATCHALL "\n" lines "${LAST_OUTPUT}")
list(LENGTH lines line_count)
if(NOT line_count EQUAL 6)
  message(FATAL_ERROR "expected header + 5 prediction rows, got:\n${LAST_OUTPUT}")
endif()

run_cli(0 verify --run run)
if(NOT LAST_OUTPUT MATCHES "bit-for-bit reproduction")
  message(FATAL_ERROR "verify did not attempt reproduction:\n${LAST_OUTPUT}")
endif()

# Tampering with the data must be detected.
file(APPEND "${WORK}/data.csv" "1,1,1,1,1,1\n")
run_cli(1 verify --run run)

# Usage errors exit with status 2.
run_cli(2 train --task regression)
run_cli(2 frobnicate)
