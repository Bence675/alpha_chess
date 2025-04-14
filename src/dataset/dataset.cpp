
#include "dataset.h"
#include "chess/chess.hpp"
#include "board_utils.h"
#include <fstream>
#include "logger.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

ChessDataSet::ChessDataSet(int max_size) : _max_size(max_size) {
    _queue = Queue<ChessData>(max_size);
}

ChessDataSet::ChessDataSet(std::vector<ChessData> data, int max_size) : _max_size(max_size), _queue(max_size) {
    
    for (const auto& item : data) {
        _queue.push(item);
    }
}

/*ChessTensorData ChessDataSet::get(size_t index) {
    if (index >= _queue.size()) {
        throw std::runtime_error("Index out of range");
    }
    auto data = _queue.pop();
    auto board = chess::Board(data.fen);
    auto input_tensor = utils::board_to_tensor(board);
    auto policy_tensor = data.policy;
    auto value_tensor = data.value;

    return {input_tensor, policy_tensor, value_tensor};

}*/

ChessTensorData ChessDataSet::get_batch(std::vector<long unsigned> indices) {
    std::vector<torch::Tensor> inputs, policies, values;
    for (auto index : indices) {
        if (index >= _queue.size()) {
            throw std::runtime_error("Index out of range");
        }
        auto data = _queue[index];
        auto board = chess::Board(data.fen);
        auto input_tensor = utils::board_to_tensor(board);
        auto policy_tensor = data.policy;
        auto value_tensor = data.value;

        inputs.push_back(input_tensor);
        policies.push_back(policy_tensor);
        values.push_back(value_tensor);
    }
    return {torch::stack(inputs), torch::stack(policies), torch::stack(values)};
}

  
torch::optional<size_t> ChessDataSet::size() const {
    return _queue.size();
}

void ChessDataSet::add_data(std::string fen, torch::Tensor policy_data, torch::Tensor labels) {
    _queue.push({fen, policy_data, labels});
}

void ChessDataSet::add_data(std::vector<ChessData> data) {
    for (const auto& item : data) {
        _queue.push(item);
    }
}

void ChessDataSet::save(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for saving");
    }
    json j;
    std::vector<std::string> fens;
    std::vector<torch::Tensor> policies, values;
    
    for (int i = 0; i < _queue.size(); ++i) {
        auto data = _queue[i];
        fens.push_back(data.fen);
        policies.push_back(data.policy);
        values.push_back(data.value);
    }
    
    j["fens"] = fens;
    
    std::ofstream fout(path + "_meta.json");
    fout << j.dump(4);
    fout.close();

    // Save tensors
    torch::save(torch::stack(policies), path + "_policies.pt");
    torch::save(torch::stack(values), path + "_values.pt");
    
    Logger::log("Saved dataset to " + path);
    Logger::log("Dataset size: " + std::to_string(_queue.size()));
    Logger::log("Dataset saved");
    ofs.close();
}

void ChessDataSet::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for loading");
    }
    _queue.clear();
    
    std::ifstream fin(path + "_meta.json");
    json j;
    fin >> j;
    fin.close();

    std::vector<std::string> fens = j["fens"];

    // Load policy and value tensors
    torch::Tensor policies, values;
    torch::load(policies, path + "_policies.pt");
    torch::load(values, path + "_values.pt");

    // Split batched tensors into individual samples
    for (size_t i = 0; i < fens.size(); ++i) {
        _queue.push({fens[i], policies[i], values[i]});
    }

    Logger::log("Loaded dataset from " + path);
    Logger::log("Dataset size: " + std::to_string(_queue.size()));
    Logger::log("Dataset loaded");
}