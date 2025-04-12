
#include "dataset.h"
#include "chess/chess.hpp"
#include "board_utils.h"

ChessDataSet::ChessDataSet(int max_size) : _max_size(max_size) {
    _queue = Queue<ChessData>(max_size);
}

ChessDataSet::ChessDataSet(std::vector<ChessData> data, int max_size) : _max_size(max_size), _queue(max_size) {
    
    for (const auto& item : data) {
        _queue.push(item);
    }
}

ChessTensorData ChessDataSet::get(size_t index) {
    if (index >= _queue.size()) {
        throw std::runtime_error("Index out of range");
    }
    auto data = _queue.pop();
    auto board = chess::Board(data.fen);
    auto input_tensor = utils::board_to_tensor(board);
    auto policy_tensor = data.policy;
    auto value_tensor = data.value;

    return {input_tensor, policy_tensor, value_tensor};

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ChessDataSet::get_batch(torch::Tensor& indices) {
    auto batch_size = indices.size(0);
    auto input_tensor = torch::empty({batch_size, 19, 8, 8});
    auto policy_tensor = torch::empty({batch_size, 73 * 64});
    auto value_tensor = torch::empty({batch_size});

    for (int i = 0; i < batch_size; i++) {
        auto data = _queue.pop();
        auto board = chess::Board(data.fen);
        input_tensor[i] = utils::board_to_tensor(board);
        policy_tensor[i] = data.policy;
        value_tensor[i] = data.value;
    }

    return std::make_tuple(input_tensor, policy_tensor, value_tensor);
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