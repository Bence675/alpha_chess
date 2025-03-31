
add_executable(
    test_trainer
    TestTrainer.cpp
)

target_link_libraries(
    test_trainer
    PUBLIC
    utils
    chess
    MCTS
    logger
    trainer
    gtest
    gtest_main
    ${TORCH_LIBRARIES}
)
  

target_include_directories(test_trainer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)


add_test(
    NAME test_trainer
    COMMAND test_trainer
)