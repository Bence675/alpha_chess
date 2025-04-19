
#include <cmath>
#include <memory>
#include <format>
#include "random.h"
#include "node.h"
#include "logger.h"
#include "string_utils.h"
#include "trainer.h"

node_t::node_t(
    chess::Board& board,
    std::shared_ptr<torch::nn::Module> model,
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
        // Logger::log(join_str(" ", "Child", std::string(child->move.from()), std::string(child->move.to()), "Value", child->value, "Visit Count", child->visit_count, "Score", score));
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
        q_value = 0.5;
    } else {
        q_value = 1 - ((child->value / child->visit_count) + 1) / 2;
    }
    // Logger::log(join_str(" ", "Child", std::string(child->move.from()), std::string(child->move.to()), "Value", child->value, "Visit Count", child->visit_count, "Q Value", q_value, "prior", child->prior, "Parent visit count", this->visit_count));
    return q_value + C_PUCT * child->prior * std::sqrt(this->visit_count) / (1 + child->visit_count);
}

float node_t::expand() {
    if (this->children.size() > 0) {
        throw std::runtime_error("Node already expanded");
    }
    if (this->board.isGameOver().second != chess::GameResult::NONE) {
        // Logger::log("expand see game over");
        return this->board.isGameOver().second == chess::GameResult::WIN ? 1.0 : 
               this->board.isGameOver().second == chess::GameResult::DRAW ? 0.0 : -1.0;
    }
    // TODO wrap the output of board_to_tensor with a torch layer
    auto state_tensor = utils::board_to_tensor(this->board);

    auto fen = split(this->board.getFen(), " ");
    std::string fen_without_fullmove;
    for(int i = 0; i < 5; i++) {
        fen_without_fullmove += fen[i] + " ";
    }
    fen_without_fullmove = fen_without_fullmove.substr(0, fen_without_fullmove.size() - 1);

    auto& memory_instance = memory::getInstance();

    while (true) {
        {
            std::unique_lock<std::mutex> lock(memory_instance.action_probs_map_mutex);
            if (memory_instance.action_probs_map.contains(fen_without_fullmove)) {
                break;
            }
        }
        // Logger::log("Waiting for model to compute action probabilities for " + fen_without_fullmove);
        
        {
            std::unique_lock<std::mutex> lock(memory_instance.boards_to_compute_and_processing_mutex);
            if (std::find(memory_instance.processing.begin(), memory_instance.processing.end(), fen_without_fullmove) == memory_instance.processing.end() &&
                std::find(memory_instance.boards_to_compute.begin(), memory_instance.boards_to_compute.end(), fen_without_fullmove) == memory_instance.boards_to_compute.end()) {
                    memory_instance.boards_to_compute.push_back(fen_without_fullmove);
            } 
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // maybe need more times
    }
    auto action_probs = memory_instance.action_probs_map[fen_without_fullmove].first;
    float sum = 0;
    for (auto& action_prob : action_probs) {
        sum += action_prob.second;
    }

    for (auto& action_prob : action_probs) {
        auto move = action_prob.first;
        auto prob = action_prob.second / sum;
        this->children.push_back(std::make_shared<node_t>(this->board, this->model, shared_from_this(), move, prob));
    }
    return memory_instance.action_probs_map[fen_without_fullmove].second;


/*
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
    memory::getInstance().action_probs_map[fen_without_fullmove] = std::make_pair(action_probs, value_tensor[0].item<float>());
    return value_tensor[0].item<float>();*/
}

void node_t::backpropagate(float value) {
    this->visit_count++;
    this->value += value;
    if (auto parent = this->parent.lock()) {
        parent->backpropagate(-value);
    }
}

chess::Move node_t::get_action() const {
    auto action_probs_raw = get_action_probs();
    std::vector<float> action_probs;

    for (int i = 0; i < action_probs_raw.size(); ++i) {
        action_probs.push_back(action_probs_raw[i].second);
    }

    return this->children[utils::random_choose(action_probs)]->move;
}

node_t::action_probs_t node_t::get_action_probs() const {
    node_t::action_probs_t action_probs;
    for (auto child : this->children) {
        action_probs.push_back(std::make_pair(child->move, static_cast<float>(child->visit_count) / this->visit_count));
    }
    return action_probs;
}

torch::Tensor node_t::get_action_probs_tensor() const {
    torch::Tensor action_probs_tensor = torch::zeros({73 * 64});
    for (auto child : this->children) {
        action_probs_tensor[utils::move_to_idx(child->move)] = static_cast<float>(child->visit_count) / this->visit_count;
        // Logger::log(join_str(" ", "Move", std::string(child->move.from()), std::string(child->move.to()), "idx", utils::move_to_idx(child->move), "Child visit count", child->visit_count, "Parent visit count", this->visit_count));
    }
    return action_probs_tensor;
}

float node_t::get_value() const {
    return this->value;
}

std::vector<std::shared_ptr<node_t>> node_t::get_children() const {
    return std::move(children);
}