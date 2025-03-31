
add_library(trainer
    trainer.cpp
)

target_link_libraries(trainer PUBLIC
    ${TORCH_LIBRARIES}
    model
    chess
    logger
    utils
    MCTS
)

target_include_directories(trainer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(test)