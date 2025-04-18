

#include "config.h"
#include <fstream>
#include <iostream>

namespace config {

Config load_config(const std::string &config_file) {
    Config config;
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_file);
    }

    nlohmann::json json_config;
    file >> json_config;


    if (json_config.contains("MCTS")) {
        config.mcts_config.load_config(json_config["MCTS"]);
    }

    if (json_config.contains("trainer")) {
        config.trainer_config.load_config(json_config["trainer"]);
    }

    if (json_config.contains("network")) {
        config.network_config.load_config(json_config["network"]);
    }

    if (json_config.contains("skip_first_self_play")) {
        config.skip_first_self_play = json_config["skip_first_self_play"].get<bool>();
    }

    if (!config.trainer_config.report_path.empty()) {
        // save config to report_path
        std::cout << "Saving config to " << config.trainer_config.report_path << "/config.json";
        std::ofstream report_file(config.trainer_config.report_path + "/config.json");
        if (!report_file.is_open()) {
            throw std::runtime_error("Could not open report file: " + config.trainer_config.report_path + "/config.json, make sure the directory exists");
        }
        report_file << json_config.dump(4); // Pretty print with 4 spaces
        report_file.close();
    }
    file.close();

    return config;
}

}