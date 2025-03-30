function(add_sanitizer_flags)
    # Check compiler
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Common sanitizers
        # set(SANITIZERS address undefined leak)
        foreach(sanitizer IN LISTS SANITIZERS)
            check_cxx_compiler_flag(-fsanitize=${sanitizer} HAVE_SANITIZER_${sanitizer})
            if(HAVE_SANITIZER_${sanitizer})
                add_compile_options(-fsanitize=${sanitizer})
                add_link_options(-fsanitize=${sanitizer})
            endif()
        endforeach()

        # For better backtraces
        if(NOT MSVC)
            add_compile_options(-fno-omit-frame-pointer)
            add_link_options(-fno-omit-frame-pointer)
        endif()
    endif()
endfunction()

enable_testing()

cmake_minimum_required(VERSION 3.20)
project("alpha_chess")

if(NOT DEFINED LIBTORCH_URL)
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip")
endif()

# Try to find LibTorch first
find_package(Torch QUIET)
if(NOT Torch_FOUND)
  message(STATUS "LibTorch not found. Downloading...")
  
  include(FetchContent)
  FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
  )
  FetchContent_MakeAvailable(libtorch)
  
  # Set the path to the downloaded LibTorch
  set(Torch_DIR "${libtorch_SOURCE_DIR}/share/cmake/Torch")
  find_package(Torch REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}" )
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pg")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -pg")

add_subdirectory(src)
add_subdirectory(include)


include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)


include(CheckCXXCompilerFlag)
add_sanitizer_flags()