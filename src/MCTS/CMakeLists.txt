
add_library(MCTS
    mcts.cpp
    node.cpp
)

target_link_libraries(MCTS PUBLIC
    ${TORCH_LIBRARIES}
    model
    chess
    config
    logger
    utils
    trainer
)

target_include_directories(MCTS
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(test)