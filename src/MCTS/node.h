
#include "chess/chess.hpp"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include "model.h"
#include "board_utils.h"


#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#define C_PUCT 1.0

class node_t : public std::enable_shared_from_this<node_t>
{

public:
    node_t(chess::Board& board, std::shared_ptr<torch::nn::Module> model,  std::shared_ptr<node_t> parent=nullptr, chess::Move move = 0, float prior=0.0);
    // node_t operator=(const node_t& node);

    // copy constructor
    /*
    node_t(const node_t& node) {
        this->parent = node.parent;
        this->move = node.move;
        this->value = node.value;
        this->visit_count = node.visit_count;
        this->is_leaf = node.is_leaf;
        this->prior = node.prior;
    }*/

    typedef std::pair<chess::Move, float> action_prob_t;
    typedef std::vector<std::pair<chess::Move, float>> action_probs_t;

    std::shared_ptr<node_t> select_best_child();
    float expand();
    void backup(float value);
    float ucb_score(const std::shared_ptr<node_t> child) const;
    std::shared_ptr<node_t> select_best_leaf();
    void backpropagate(float value);
    chess::Move get_action() const;
    float get_value() const;
    action_probs_t get_action_probs() const;
    torch::Tensor get_action_probs_tensor() const;

    chess::Board& board;
    std::shared_ptr<torch::nn::Module> model;
    std::weak_ptr<node_t> parent;
    chess::Move move;
    std::vector<std::shared_ptr<node_t>> children;
    float value;
    int visit_count;
    float prior;
};

#endif // MCTS_NODE_H
