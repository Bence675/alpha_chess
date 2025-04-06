

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "chess/chess.hpp"
#include "model.h"
#include "logger.h"
#include "node.h"
#include "string_utils.h"





class KotHModel : public Model {
public:
    KotHModel() {}

    void _check_input(torch::Tensor x) override {
        if (x.sizes().size() != 4) {
            throw std::runtime_error("Input tensor must have 4 dimensions");
        }
        if (x.sizes() != torch::IntArrayRef({x.sizes()[0], 19, 8, 8})) {
            std::string actual_str = "[";
            for (int i = 0; i < x.sizes().size(); i++) {
                actual_str += std::to_string(x.sizes()[i]);
                if (i < x.sizes().size() - 1) {
                    actual_str += ", ";
                }
            }
            actual_str += "]";
            throw std::runtime_error(join_str(" ", "Expected dimension: [19, 8, 8], actual:", actual_str));

        }
    }

    void _check_output(std::tuple<torch::Tensor, torch::Tensor> output) {
        auto policy = std::get<0>(output);
        auto value = std::get<1>(output);
        if (policy.sizes() != torch::IntArrayRef({policy.sizes()[0], 73 * 64})) {
            std::string actual_str = "[";
            for (int i = 0; i < policy.sizes().size(); i++) {
                actual_str += std::to_string(policy.sizes()[i]);
                if (i < policy.sizes().size() - 1) {
                    actual_str += ", ";
                }
            }
            actual_str += "]";
            throw std::runtime_error(join_str(" ", "Expected dimension: [73 * 64], actual:", actual_str));
        }
        if (value.sizes() != torch::IntArrayRef({value.sizes()[0], 1})) {
            std::string actual_str = "[";
            for (int i = 0; i < value.sizes().size(); i++) {
                actual_str += std::to_string(value.sizes()[i]);
                if (i < value.sizes().size() - 1) {
                    actual_str += ", ";
                }
            }
            actual_str += "]";
            throw std::runtime_error(join_str(" ", "Expected dimension: [1], actual:", actual_str));
        }
    }
    

    std::tuple<torch::Tensor, torch::Tensor> forward_single(torch::Tensor x) {
        // Logger::log("Forward pass KotHModel");
        if (x.sizes() != torch::IntArrayRef({19, 8, 8})) {
            std::string actual_str = "[";
            for (int i = 0; i < x.sizes().size(); i++) {
                actual_str += std::to_string(x.sizes()[i]);
                if (i < x.sizes().size() - 1) {
                    actual_str += ", ";
                }
            }
            actual_str += "]";
            throw std::runtime_error("Expected dimension: [19, 8, 8], actual: " + actual_str);
        }
        int action_space = 73 * 64;
        bool is_white = x[17][0][0].item<float>();
        chess::Square king_pos;
        chess::Square enemy_king_pos;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                for (int k : {5, 11}) {
                    if (x[k][i][j].item<float>() == 1) {
                        ((k == 5) == is_white ? king_pos : enemy_king_pos) = chess::Square(chess::File(i), chess::Rank(j));
                    }
                }
            }
        }
        // Logger::log("Middle KotHModel");
        if (king_pos == chess::Square::NO_SQ || enemy_king_pos == chess::Square::NO_SQ) {
            throw std::runtime_error("King not found");
        }
        auto king_file = int(king_pos.file());
        auto king_rank = int(king_pos.rank());
        auto enemy_king_file = int(enemy_king_pos.file());
        auto enemy_king_rank = int(enemy_king_pos.rank());
                

        auto policy = torch::zeros({action_space});
        for (int i = 0; i < action_space; i++) {
            // Logger::log(join_str(" ", "Action", i));
            auto opt_move = utils::idx_to_move(i);
            if (!opt_move.has_value()) {
                continue;
            }
            auto move = opt_move.value();
            auto from_square = move.from();
            auto to_square = move.to();


            int from_rank = int(from_square.rank());
            int from_file = int(from_square.file());

            int to_rank = int(to_square.rank());
            int to_file = int(to_square.file());

            
            // Logger::log(join_str(" ", "From", from_rank, from_file));
            // Logger::log(join_str(" ", "To", to_rank, to_file));


            auto current_piece_distance_from_center = std::abs(from_file - 3.5) + std::abs(from_rank - 3.5);
            auto new_piece_distance_from_center = std::abs(to_file - 3.5) + std::abs(to_rank - 3.5);

            // move a piece to the center
            for (int j = 0; j < 6; ++j) {
                if (x[is_white ? j : (j + 6)][from_rank][from_file].item<float>() == 1) {
                    policy[i] += 
                                (current_piece_distance_from_center - new_piece_distance_from_center) * // delta distance from center 
                                (j + 1); // piece value

                    // Logger::log(join_str(" ", "Piece move", policy[i].item<float>()));
                }
            }

            // take a piece
            for (int j = 0; j < 6; ++j) {
                if (x[is_white ? j : (j + 6)][to_rank][to_file].item<float>() == 1) {
                    policy[i] += (j + 1) * 10;
                    // Logger::log(join_str(" ", "Piece take", policy[i].item<float>()));
                }
            }

            // move king closer to center
            if (x[(is_white ? 5 : 11)][from_rank][from_file].item<float>() == 1) {
                policy[(is_white ? 5 : 11)] += (current_piece_distance_from_center - new_piece_distance_from_center) * 10;
                // Logger::log(join_str(" ", "King move", policy[(is_white ? 5 : 11)].item<float>()));
            }
        }
        
        // Logger::log("Almost end pass KotHModel");
        auto value = torch::zeros({1});
        auto king_distance = std::abs(king_file - 3.5) + std::abs(king_rank - 3.5);
        auto enemy_king_distance = std::abs(enemy_king_file - 3.5) + std::abs(enemy_king_rank - 3.5);
        value[0] = (king_distance - enemy_king_distance);
        // Logger::log("End forward pass KotHModel");
        return std::make_tuple(policy, value);
    }
        

    std::tuple<torch::Tensor, torch::Tensor> _forward(torch::Tensor x) override {
        if (x.sizes().size() != 4) {
            throw std::runtime_error("Input tensor must have 4 dimensions");
        }
        int action_space = 73 * 64;
        auto res_policy_value = torch::zeros({x.sizes()[0], action_space});
        auto res_value = torch::zeros({x.sizes()[0], 1});

        for (int i = 0; i < x.sizes()[0]; i++) {
            auto [policy, value] = forward_single(x[i]);
            res_policy_value[i] = policy;
            res_value[i] = value;
        }
        return std::make_tuple(res_policy_value, res_value);
    }
};