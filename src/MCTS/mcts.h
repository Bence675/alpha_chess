
#include <memory>
#include <torch/torch.h>
#include <vector>
#include "model.h"
#include "node.h"
#include "chess/chess.hpp"
#include "config.h"



#ifndef MCTS_H
#define MCTS_H

struct HistoryObject
{
    torch::Tensor state;
    torch::Tensor action_prob;
    float value;
    chess::GameResult result;

    HistoryObject(torch::Tensor state, torch::Tensor action_prob, float value, chess::GameResult result) : state(state), action_prob(action_prob), value(value), result(result) {}
};

class MCTS
{
private:
    unsigned int num_simulations;
    float c_puct;
public:
    MCTS(std::shared_ptr<torch::nn::Module> model, const config::Config::MCTSConfig& config);
    std::shared_ptr<node_t> search(const chess::Board& board, int iteration = 0);
    void simulate(std::shared_ptr<node_t> root);
    std::vector<std::pair<chess::Move, float>> _compute_policy(node_t& node);
    void set_model(std::shared_ptr<torch::nn::Module> model);
    ~MCTS();
    std::shared_ptr<torch::nn::Module> model;

};

#endif // MCTS_H