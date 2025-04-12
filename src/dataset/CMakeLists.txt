

add_library(
    dataset
    dataset.cpp
)

target_link_libraries(dataset PUBLIC
    ${TORCH_LIBRARIES}
    chess
    utils
)

target_include_directories(dataset
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
