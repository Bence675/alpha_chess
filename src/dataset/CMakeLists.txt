

add_library(
    dataset
    dataset.cpp
)

target_link_libraries(dataset PUBLIC
    ${TORCH_LIBRARIES}
    chess
    json
    logger
    utils
)

target_include_directories(dataset
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
