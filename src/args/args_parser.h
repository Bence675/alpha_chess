
#include <string>

#ifndef ALPHA_CHESS_ARGS_PARSER_H
#define ALPHA_CHESS_ARGS_PARSER_H

namespace args {

struct args {
    bool help = false;
    std::string config_file = "";
    std::string report_output = "";
};

args parse_args(int argc, char *argv[]);
void print_help();

}

#endif //ALPHA_CHESS_ARGS_PARSER_H