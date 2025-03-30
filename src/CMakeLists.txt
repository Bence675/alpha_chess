

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(logger)
add_subdirectory(model)
add_subdirectory(MCTS)
add_subdirectory(utils)

add_executable(
    alpha_chess
    main.cpp
)

target_link_libraries(
    alpha_chess
    PUBLIC
    logger
    model
    MCTS
    utils
    chess
    ${TORCH_LIBRARIES}
)