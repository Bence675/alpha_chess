

add_library(model
    model.cpp
)

target_include_directories(model
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(model
    PUBLIC
    ${TORCH_LIBRARIES}
    logger
    config
)
