

add_executable(
    test_utils
    TestUtils.cpp
    TestStringUtils.cpp
)

target_link_libraries(
    test_utils
    PUBLIC
    utils
    chess
    logger
    gtest
    gtest_main
    ${TORCH_LIBRARIES}
)
  

target_include_directories(
    test_utils
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)


add_test(
    NAME test_utils
    COMMAND test_utils
)