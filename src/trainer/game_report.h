
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Child {
    std::string action;
    float prob;
    float value;
    int visit_count;
    float prior;

    json to_json() const {
        json j;
        j["action"] = action;
        j["prob"] = prob;
        j["value"] = value;
        j["visit_count"] = visit_count;
        j["prior"] = prior;
        return j;
    }
};

struct MoveReport {
    std::string fen;
    std::string move;
    std::vector<Child> children;
    float value;

    json to_json() const {
        json j;
        j["fen"] = fen;
        j["move"] = move;
        j["children"] = json::array();
        for (const auto& child : children) {
            j["children"].push_back(child.to_json());
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