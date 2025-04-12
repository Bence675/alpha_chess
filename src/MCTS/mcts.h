
#include <memory>
#include <torch/torch.h>
#include <vector>
#include "model.h"
#include "node.h"
#include "chess/chess.hpp"



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
    float epsilon;
public:
    MCTS(std::shared_ptr<torch::nn::Module> model, unsigned int num_simulations, float c_puct, float epsilon);
    std::shared_ptr<node_t> search(const chess::Board& board);
    void simulate(std::shared_ptr<node_t> root);
    std::vector<std::pair<chess::Move, float>> _compute_policy(node_t& node);
    void set_model(std::shared_ptr<torch::nn::Module> model);
    ~MCTS();
    std::shared_ptr<torch::nn::Module> model;

};

#endif // MCTS_H