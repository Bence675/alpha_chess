
#include <nlohmann/json.hpp>
#include <fstream>

#ifndef CONFIG_H
#define CONFIG_H

namespace config {

template <typename T>
inline
T lookup(const nlohmann::json &json_config, const std::string &key, const T &default_value) {
    if (json_config.contains(key)) {
        return json_config[key].get<T>();
    }
    return default_value;
}

struct Config {

    struct MCTSConfig {
        int num_simulations = 100;
        float exploration_constant = 1.0;

        void load_config(const nlohmann::json &json_config) {
            num_simulations = lookup(json_config, "num_simulations", num_simulations);
            exploration_constant = lookup(json_config, "exploration_constant", exploration_constant);
        }
    };
    
    struct NetworkConfig {
        int64_t in_channels;
        int64_t num_blocks;
        int64_t num_filters;
        int64_t policy_head_filters;
        int64_t value_head_filters;
        int64_t reduction_ratio;

        void load_config(const nlohmann::json &json_config) {
            in_channels = 19;
            num_blocks = lookup(json_config, "num_blocks", 20);
            num_filters = lookup(json_config, "num_filters", 256);
            policy_head_filters = lookup(json_config, "policy_head_filters", 256);
            value_head_filters = lookup(json_config, "value_head_filters", 256);
            reduction_ratio = lookup(json_config, "reduction_ratio", 16);
        }
    };
    
    struct TrainerConfig {
    
        struct SelfPlayConfig {
            int num_iterations = 1000;
            int num_games_per_iteration = 1024;
            int max_threads = 1024;

            void load_config(const nlohmann::json &json_config) {
                num_iterations = lookup(json_config, "num_iterations", num_iterations);
                num_games_per_iteration = lookup(json_config, "num_games_per_iteration", num_games_per_iteration);
                max_threads = lookup(json_config, "max_threads", max_threads);
            }
        };
        
        struct TrainingConfig {
            int num_epochs = 10;
            int batch_size = 32;

            void load_config(const nlohmann::json &json_config) {
                num_epochs = lookup(json_config, "num_epochs", num_epochs);
                batch_size = lookup(json_config, "batch_size", batch_size);
            }
        };
    
        SelfPlayConfig self_play_config;
        TrainingConfig training_config;

        void load_config(const nlohmann::json &json_config) {
            if (json_config.contains("self_play")) {
                self_play_config.load_config(json_config["self_play"]);
            }
            if (json_config.contains("training")) {
                training_config.load_config(json_config["training"]);
            }
        }
    };

    MCTSConfig mcts_config;
    TrainerConfig trainer_config;
    NetworkConfig network_config;
};

Config load_config(const std::string& config_file);

} // namespace config

#endif // CONFIG_H