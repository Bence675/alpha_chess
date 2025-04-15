


add_library(
    args
    args_parser.cpp
)

target_link_libraries(args
    PUBLIC
    logger
)

target_include_directories(args
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)