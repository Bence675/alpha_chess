

#include "config.h"

namespace config {

Config load_config(const std::string &config_file) {
    Config config;
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_file);
    }

    nlohmann::json json_config;
    file >> json_config;

    file.close();

    if (json_config.contains("mcts")) {
        config.mcts_config.load_config(json_config["mcts"]);
    }

    if (json_config.contains("trainer")) {
        config.trainer_config.load_config(json_config["trainer"]);
    }

    if (json_config.contains("network")) {
        config.network_config.load_config(json_config["network"]);
    }

    return config;
}

}