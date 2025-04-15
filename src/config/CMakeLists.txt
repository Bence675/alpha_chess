


add_library(
    config
    config.cpp
)

target_link_libraries(config
    PUBLIC
    chess
)

target_include_directories(config
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
