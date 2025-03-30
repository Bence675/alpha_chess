

add_executable(
    test_mcts
    TestMCTS.cpp
    TestNode.cpp
)

target_link_libraries(
    test_mcts
    PUBLIC
    utils
    chess
    MCTS
    logger
    gtest
    gtest_main
    ${TORCH_LIBRARIES}
)
  

target_include_directories(test_mcts
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)


add_test(
    NAME test_mcts
    COMMAND test_mcts
)