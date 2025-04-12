

add_library(
    utils
    board_utils.cpp
)

target_include_directories(utils
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(utils PUBLIC
    ${TORCH_LIBRARIES}
    chess
    logger
    MCTS
)

add_subdirectory(test)
