cmake_minimum_required(VERSION 3.23)
project(gpt2cpp VERSION 0.1.0 LANGUAGES CXX)

option(GPT2CPP_ENABLE_TORCH "Enable LibTorch-backed model/training/inference" ON)
option(GPT2CPP_ENABLE_CUDA "Enable CUDA device selection where supported" ON)
option(GPT2CPP_BUILD_TESTS "Build tests" ON)
option(GPT2CPP_WARNINGS_AS_ERRORS "Warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/CompilerWarnings.cmake)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

add_library(gpt2cpp_core
  src/core/files.cpp
  src/core/logging.cpp
  src/core/config.cpp
  src/core/device.cpp
  src/tokenizer/bpe_tokenizer.cpp
  src/data/dataset.cpp
  src/cli/command_dispatch.cpp
)
target_include_directories(gpt2cpp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpt2cpp_core PUBLIC Threads::Threads)
gpt2cpp_set_project_warnings(gpt2cpp_core ${GPT2CPP_WARNINGS_AS_ERRORS})

set(GPT2CPP_HAVE_TORCH OFF)
if (GPT2CPP_ENABLE_TORCH)
  find_package(Torch QUIET)
  if (Torch_FOUND)
    set(GPT2CPP_HAVE_TORCH ON)
    target_link_libraries(gpt2cpp_core PUBLIC "${TORCH_LIBRARIES}")
    target_compile_definitions(gpt2cpp_core PUBLIC GPT2CPP_HAVE_TORCH=1 $<$<BOOL:${GPT2CPP_ENABLE_CUDA}>:GPT2CPP_ENABLE_CUDA=1>)
    add_library(gpt2cpp_llm
      src/model/gpt2_model.cpp
      src/training/trainer.cpp
      src/inference/generator.cpp
      src/app/chat_repl.cpp
      src/serving/http_server.cpp
    )
    target_include_directories(gpt2cpp_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gpt2cpp_llm PUBLIC gpt2cpp_core "${TORCH_LIBRARIES}" Threads::Threads)
    target_compile_definitions(gpt2cpp_llm PUBLIC GPT2CPP_HAVE_TORCH=1 $<$<BOOL:${GPT2CPP_ENABLE_CUDA}>:GPT2CPP_ENABLE_CUDA=1>)
    gpt2cpp_set_project_warnings(gpt2cpp_llm ${GPT2CPP_WARNINGS_AS_ERRORS})
  else()
    message(WARNING "LibTorch not found. Building tokenizer/data/CLI-only subset.")
  endif()
endif()

add_executable(gpt2cpp tools/gpt2cpp.cpp)
target_link_libraries(gpt2cpp PRIVATE gpt2cpp_core)
if (GPT2CPP_HAVE_TORCH)
  target_link_libraries(gpt2cpp PRIVATE gpt2cpp_llm)
endif()

foreach(alias_target IN ITEMS train_tokenizer prepare_data train_model eval_model generate chat_cli serve_web)
  add_executable(${alias_target} tools/${alias_target}.cpp)
  target_link_libraries(${alias_target} PRIVATE gpt2cpp_core)
  if (GPT2CPP_HAVE_TORCH)
    target_link_libraries(${alias_target} PRIVATE gpt2cpp_llm)
  endif()
endforeach()

if (GPT2CPP_BUILD_TESTS)
  enable_testing()
  add_executable(test_bpe_roundtrip tests/tokenizer/test_bpe_roundtrip.cpp)
  target_link_libraries(test_bpe_roundtrip PRIVATE gpt2cpp_core)
  add_test(NAME tokenizer_roundtrip COMMAND test_bpe_roundtrip)

  add_executable(test_packed_dataset tests/data/test_packed_dataset.cpp)
  target_link_libraries(test_packed_dataset PRIVATE gpt2cpp_core)
  add_test(NAME packed_dataset COMMAND test_packed_dataset)

  if (GPT2CPP_HAVE_TORCH)
    add_executable(test_model_shapes tests/model/test_model_shapes.cpp)
    target_link_libraries(test_model_shapes PRIVATE gpt2cpp_llm)
    add_test(NAME model_shapes COMMAND test_model_shapes)
  endif()
endif()
