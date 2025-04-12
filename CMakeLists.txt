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

cmake_minimum_required(VERSION 3.20)
project(alpha_chess)

enable_testing()

include_directories(/usr/local/cuda/include/nvtx3)

# Torch configuration

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}" )
set(CMAKE_CXX_STANDARD 20)


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
# add_sanitizer_flags()