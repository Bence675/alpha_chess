
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

class ChessDataSet : public torch::data::Dataset<ChessDataSet, ChessTensorData> {
public:
    ChessDataSet(int max_size = 0);
    ChessDataSet(std::vector<ChessData> data, int max_size = 0);

    // Return a single example from the dataset
    ChessTensorData get(size_t index) override;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_batch(torch::Tensor& indices);

    // Return the size of the dataset
    torch::optional<size_t> size() const override;

    void add_data(std::string fen, torch::Tensor value, torch::Tensor policy);
    void add_data(std::vector<ChessData> data);

private:
    Queue<ChessData> _queue;
    const int _max_size;
};