cmake_minimum_required(VERSION 3.21)
project ("bpe.cpp" CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
find_package(ICU REQUIRED COMPONENTS uc i18n)

add_library(bpecpp bpe.cpp bpe.h)
target_compile_features(bpecpp PUBLIC cxx_std_11)
target_link_libraries(bpecpp PUBLIC ICU::i18n ICU::uc)

add_executable(testtok testtok.cpp)
target_link_libraries(testtok PRIVATE bpecpp)