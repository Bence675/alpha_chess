

add_library(
    logger
    logger.cpp
)


target_link_libraries(logger
    PUBLIC
    utils
)
target_include_directories(logger
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
