
#include <torch/torch.h>
#include <mutex>
#include <queue>
#include "queue.h"

struct ChessData {
    std::string fen;
    torch::Tensor policy;
    torch::Tensor value;
};

struct ChessTensorData {
    torch::Tensor input;
    torch::Tensor policy;
    torch::Tensor value;
};

/*
struct CollateChessData {
    using OutputBatchType = ChessTensorData;
    ChessTensorData operator()(std::vector<ChessTensorData> samples) {
        std::vector<torch::Tensor> inputs, policies, values;
        for (auto& sample : samples) {
            inputs.push_back(sample.input);
            policies.push_back(sample.policy);
            values.push_back(sample.value);
        }
        return {
            torch::stack(inputs),    // Stack inputs along batch dimension
            torch::stack(policies),  // Stack policies
            torch::stack(values)     // Stack values
        };
    }
};*/

class ChessDataSet : public torch::data::BatchDataset<ChessDataSet, ChessTensorData, std::vector<long unsigned>> {
public:
    ChessDataSet(int max_size = 0);
    ChessDataSet(std::vector<ChessData> data, int max_size = 0);

    // Return a single example from the dataset
    // ChessTensorData get(size_t index) override;
    ChessTensorData get_batch(std::vector<long unsigned> indices) override;

    // Return the size of the dataset
    torch::optional<size_t> size() const override;

    void add_data(std::string fen, torch::Tensor value, torch::Tensor policy);
    void add_data(std::vector<ChessData> data);

    ChessDataSet(const ChessDataSet& other) : _max_size(other._max_size) {
        // Copy constructor
        _queue = std::move(other._queue);
    }
    ChessDataSet& operator=(const ChessDataSet&) = default;
    ChessDataSet(ChessDataSet&& other) : _max_size(other._max_size) {
        _queue = std::move(other._queue);
    }
    ChessDataSet& operator=(ChessDataSet&& other) = default;

    void save(const std::string& path) const;

    void load(const std::string& path);

    void clear() {
        _queue.clear();
    }

private:
    Queue<ChessData> _queue;
    const int _max_size;
};