
#include "args_parser.h"
#include "logger.h"


namespace args {

args parse_args(int argc, char *argv[]) {
    args parsed_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            parsed_args.help = true;
        } else if (arg == "--config" || arg == "-c") {
            if (i + 1 < argc) {
                parsed_args.config_file = argv[++i];
            } else {
                Logger::log("Error: --config option requires an argument.");
                parsed_args.help = true;
            }
        }
    }
    return parsed_args;
}
void print_help() {
    Logger::log("Usage: program [options]");
    Logger::log("Options:");
    Logger::log("  --help, -h       Show this help message and exit");
    Logger::log("  --config, -c     Specify the configuration file");
}

} // namespace args