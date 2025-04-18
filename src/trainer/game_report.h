
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct ActionProb {
    std::string action;
    float prob;

    json to_json() const {
        json j;
        j["action"] = action;
        j["prob"] = prob;
        return j;
    }
};

struct MoveReport {
    std::string fen;
    std::string move;
    std::vector<ActionProb> action_probs;
    float value;

    json to_json() const {
        json j;
        j["fen"] = fen;
        j["move"] = move;
        j["action_probs"] = json::array();
        for (const auto& action_prob : action_probs) {
            j["action_probs"].push_back(action_prob.to_json());
        }
        j["value"] = value;
        return j;
    }
};

struct GameReport {
    std::vector<MoveReport> moves;
    std::string result;

    json to_json() const {
        json j;
        j["moves"] = json::array();
        for (const auto& move : moves) {
            j["moves"].push_back(move.to_json());
        }
        j["result"] = result;
        return j;
    }

    void save(const std::string& path) const {
        json j = to_json();
        std::ofstream file(path);
        if (file.is_open()) {
            file << j.dump(4); // Pretty print with 4 spaces
            file.close();
        } else {
            throw std::runtime_error("Could not open file: " + path);
        }
    }
};