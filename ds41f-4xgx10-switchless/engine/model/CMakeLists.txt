add_library(dsmodel STATIC model.cu ops.cu fused.cu sample.cu prefill.cu state.cu spec.cu wave_probe.cu spill.cpp
  glp.cpp glp.cu)
target_include_directories(dsmodel PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(dsmodel PUBLIC dsattn dsk_kernels dsgb10_comm engram dsw_loader dspark)

add_executable(coll_tests tests/coll_tests.cu)
target_include_directories(coll_tests PRIVATE ${CMAKE_SOURCE_DIR}/attn/tests)
target_link_libraries(coll_tests dsmodel)

# GLP (model/glp.h): the reader against a real file and the projection's arithmetic. Host only, no GPU: a
# GGUF file's metadata, the fail-closed refusals, and glp_project_row against a reference fold.
add_executable(glp_test tests/glp_test.cpp glp.cpp)
target_include_directories(glp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Host-only recorded-Owen v1 fold/reduction-order regression (real direction file argument).
add_executable(glp_math_test tests/glp_math_test.cpp glp.cpp)
target_include_directories(glp_math_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# B3: the batch spec step's layout and verdicts (spec_batch.h) against a host reference; host only, no GPU
add_executable(spec_batch_test tests/spec_batch_test.cu)
target_include_directories(spec_batch_test PRIVATE ${CMAKE_SOURCE_DIR})

# Profiling builds (-DDSGB10_NVTX=ON): NVTX ranges per turn, chunk, layer and part for nsys (model/nvtx.h),
# and DSGB10_ROUTE_DUMP (model/prefill.cu). The ranges cost nothing without a profiler attached.
option(DSGB10_NVTX "NVTX ranges in the model and the server, for nsys" OFF)
if(DSGB10_NVTX)
  target_compile_definitions(dsmodel PUBLIC DSGB10_NVTX)
endif()
