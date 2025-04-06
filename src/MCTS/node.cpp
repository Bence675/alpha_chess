
#include <cmath>
#include <memory>
#include <format>
#include "node.h"
#include "logger.h"
#include "string_utils.h"
#include "random.h"

node_t::node_t(
    chess::Board& board,
    std::shared_ptr<Model> model,
    std::shared_ptr<node_t> parent,
    chess::Move move,
    float prior
) : board(board),
    model(model),
    parent(parent),
    move(move),
    value(0.0),
    visit_count(0),
    prior(prior) {}

/*
node_t node_t::operator=(const node_t& node) {
    this->parent = node.parent;
    this->move = node.move;
    this->value = node.value;
    this->visit_count = node.visit_count;
    this->prior = node.prior;
    return *this;
}
*/

std::shared_ptr<node_t> node_t::select_best_child() {
    float best_score = std::numeric_limits<int>::min();
    std::shared_ptr<node_t> best_child;
    for(auto child : this->children) {
        float score = this->ucb_score(child);
        if(score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    return best_child;
}

std::shared_ptr<node_t> node_t::select_best_leaf() {
    auto node = shared_from_this();
    while (node->children.size() > 0) {
        if (node->board.isGameOver().second != chess::GameResult::NONE) {
            Logger::log("Maybe segfault saver1");
            break;
        }
        node = node->select_best_child();
        if (!node) {
            Logger::log("Maybe segfault saver");
            break;
        }
        // Logger::log(join_str(" ", "Node", std::string(node->move.from()), std::string(node->move.to()), "Value", node->value, "Visit Count", node->visit_count));
        // Logger::log("Fen: " + node->board.getFen());
        // Logger::log("Move: " + std::string(node->move.from()) + " " + std::string(node->move.to()));
        node->board.makeMove(node->move);
    }
    return node;
}

float node_t::ucb_score(const std::shared_ptr<node_t> child) const {
    float q_value;
    if (child->visit_count == 0) {
        q_value = 0;
    } else {
        q_value = 1 - ((child->value / child->visit_count) + 1) / 2;
    }
    return q_value + C_PUCT * child->prior * std::sqrt(this->visit_count) / (1 + child->visit_count);
}

float node_t::expand() {
    if (this->children.size() > 0) {
        throw std::runtime_error("Node already expanded");
    }
    if (this->board.isGameOver().second != chess::GameResult::NONE) {
        return 1 ? this->board.isGameOver().second == chess::GameResult::WIN :
              -1 ? this->board.isGameOver().second == chess::GameResult::LOSE : 
               0; 
    }
    // TODO wrap the output of board_to_tensor with a torch layer
    auto state_tensor = utils::board_to_tensor(this->board);

    auto fen = split(this->board.getFen(), " ");
    std::string fen_without_fullmove;
    for(int i = 0; i < 5; i++) {
        fen_without_fullmove += fen[i] + " ";
    }
    fen_without_fullmove = fen_without_fullmove.substr(0, fen_without_fullmove.size() - 1);

    if (node_t::action_probs_map.contains(fen_without_fullmove)) {
        auto action_probs = node_t::action_probs_map[fen_without_fullmove].first;
        for (auto& action_prob : action_probs) {
            auto move = action_prob.first;
            auto prob = action_prob.second;
            this->children.push_back(std::make_shared<node_t>(this->board, this->model, shared_from_this(), move, prob));
        }
        return node_t::action_probs_map[fen_without_fullmove].second;
    }

    auto output = this->model->forward(torch::unsqueeze(state_tensor, 0));

    auto policy_tensor = std::get<0>(output)[0];
    auto value_tensor = std::get<1>(output)[0];
    chess::Movelist legal_moves;
    chess::movegen::legalmoves(legal_moves, this->board);
    action_probs_t action_probs;

    for (auto move : legal_moves) {
        auto idx = utils::move_to_idx(move);
        //Logger::log(join_str(" ", "Move", std::string(move.from()), std::string(move.to()), "idx", idx));
        // Logger::log(join_str(" ", "Move", std::string(move.from()), std::string(move.to()), "idx", idx));
        this->children.push_back(std::make_shared<node_t>(this->board, this->model, shared_from_this(), move, policy_tensor[idx].item<float>()));
        action_probs.push_back(std::make_pair(move, policy_tensor[idx].item<float>()));
    }
    node_t::action_probs_map[fen_without_fullmove] = std::make_pair(action_probs, value_tensor[0].item<float>());
    return value_tensor[0].item<float>();
}

void node_t::backpropagate(float value) {
    this->visit_count++;
    this->value += value;
    if (auto parent = this->parent.lock()) {
        parent->backpropagate(-value);
    }
}

chess::Move node_t::get_action() const {
    std::vector<float> action_probs(this->children.size());
    for (size_t i = 0; i < this->children.size(); ++i) {
        action_probs[i] = this->children[i]->prior;
    }

    return this->children[utils::random_choose(action_probs)]->move;
}