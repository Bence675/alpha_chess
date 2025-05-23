

#include "trainer.h"
#include "node.h"
#include "mcts.h"
#include "model.h"
#include "string_utils.h"
#include "thread_pool.h"
#include "game_report.h"


Trainer::Trainer(const config::Config& config) : config(config.trainer_config) {
    Logger::log("Trainer constructor");
    _model = std::make_shared<LCZero>(config.network_config);
    _model->to(torch::kCUDA);
    _model->eval();
    _optimizer = std::make_shared<torch::optim::Adam>(_model->parameters(), torch::optim::AdamOptions(0.001));
    Logger::log("Model created");
    _mcts = std::make_shared<MCTS>(_model, config.mcts_config);
    // auto dataset = ChessDataSet(1000000).map(torch::data::transforms::Stack<>());
    // _dataset = ChessDataSet(1000000).map(torch::data::transforms::Stack<>());
    

    model_thread = std::thread([this]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // if (!_self_playing) {
            //     Logger::log("Waiting forself play");
             //    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            //     continue;
            // }
            torch::Tensor input_tensor;
            
            {
                std::unique_lock<std::mutex> lock(memory::getInstance().boards_to_compute_and_processing_mutex);
                if (memory::getInstance().boards_to_compute.size() == 0) {
                    continue;
                }
                memory::getInstance().processing = memory::getInstance().boards_to_compute;
                // Logger::log("Processing " + std::to_string(memory::getInstance().processing.size()) + " boards");
                input_tensor = torch::zeros({static_cast<long>(memory::getInstance().boards_to_compute.size()), 19, 8, 8});
                // Logger::log("Transforming boards to tensor");
                for (int i = 0; i < memory::getInstance().boards_to_compute.size(); i++) {
                    auto board = chess::Board(memory::getInstance().boards_to_compute[i]);
                    input_tensor[i] = utils::board_to_tensor(board);
                }
                // Logger::log("Tensor created");
                memory::getInstance().boards_to_compute.clear();
            }
            if (input_tensor.sizes()[0] == 0) {
                Logger::log("No boards to compute");
                continue;
            }
            // Logger::log("Computing action probabilities for " + std::to_string(input_tensor.sizes()[0]) + " boards");
            auto output = _model->forward(input_tensor);
            auto policy_tensor = std::get<0>(output).to(torch::kCPU);
            auto value_tensor = std::get<1>(output).to(torch::kCPU);

            // Logger::log("Model called");

            auto tmp_action_probs_map = std::unordered_map<std::string, std::pair<node_t::action_probs_t, float>>();

            for (int i = 0; i < memory::getInstance().processing.size(); i++) {
                auto fen_without_fullmove = memory::getInstance().processing[i];
                auto board = chess::Board(fen_without_fullmove + " 0");
                auto action_probs = std::vector<std::pair<chess::Move, float>>();
                auto legal_moves = chess::Movelist();
                chess::movegen::legalmoves(legal_moves, board);

                for (auto move : legal_moves) {
                    auto idx = utils::move_to_idx(move);
                    action_probs.push_back(std::make_pair(move, policy_tensor[i][idx].item<float>()));
                }

                auto legal_action_probs = std::vector<float>();
                for (auto move : legal_moves) {
                    auto idx = utils::move_to_idx(move);
                    legal_action_probs.push_back(policy_tensor[i][idx].item<float>());
                }

                auto valid_action_probs_tensor = torch::tensor(legal_action_probs)
                    .view({1, static_cast<long>(legal_action_probs.size())});
                auto action_probs_tensor = torch::nn::functional::softmax(
                    valid_action_probs_tensor,
                    torch::nn::functional::SoftmaxFuncOptions(1).dtype(torch::kFloat32)
                );

                for (int j = 0; j < action_probs.size(); j++) {
                    action_probs[j].second = action_probs_tensor[0][j].item<float>();
                }
                tmp_action_probs_map[fen_without_fullmove] = std::make_pair(std::move(action_probs), value_tensor[i].item<float>());
            }
            // Logger::log("Between");

            {   
                // Logger::log("Waiting for action_probs_map_mutex");
                std::unique_lock<std::mutex> lock(memory::getInstance().action_probs_map_mutex);
                // Logger::log("Muterx locked");
                for (auto& item : tmp_action_probs_map) {
                    memory::getInstance().action_probs_map[item.first] = item.second;
                }
                // Logger::log("Mutex unlocked");
            }

            /*{
                std::unique_lock<std::mutex> lock(memory::getInstance().action_probs_map_mutex);
                Logger::log("Action probs map size: " + std::to_string(memory::getInstance().action_probs_map.size()));
            }*/

            {
                // Logger::log("Waiting for boards_to_compute_and_processing_mutex");
                std::unique_lock<std::mutex> lock(memory::getInstance().boards_to_compute_and_processing_mutex);
                memory::getInstance().processing.clear();
                // Logger::log("Processing cleared");
            }
        }
    });
    model_thread.detach();
}

void Trainer::self_play(int iteration) {
    _self_playing = true;
    _dataset.clear();

    auto& trainer_config = config.self_play_config;
    ThreadPool pool(trainer_config.max_threads);
    

    torch::Tensor data_memory;
    torch::Tensor target_memory;
    for (int i = 0; i < trainer_config.num_games_per_iteration; i++) {
        Logger::log("Game " + std::to_string(i));
        pool.enqueue([this, iteration, i]() {
            play_game(iteration, i);
        });
    }

    pool.wait();

    // save dataset
}

void Trainer::play_game(int iteration, int game) {
    chess::Board board;
    _self_playing = false;
    std::vector<ChessData> history;
    GameReport game_report;
    while (true) {
        Logger::log("Cache size: " + std::to_string(memory::getInstance().action_probs_map.size()));
        Logger::log("Current Board: " + board.getFen());
        auto root = _mcts->search(board, iteration);
        // Logger::log("Search");
        auto action = root->get_action();
        // Logger::log("Action: " + to_string(action));
        
        history.push_back(ChessData{board.getFen(), root->get_action_probs_tensor(), torch::zeros({1})});
        
        MoveReport move_report;
        move_report.fen = board.getFen();
        
        board.makeMove(action);

        // Report the move
        move_report.move = to_string(action);
        auto action_probs = root->get_action_probs();
        auto children = root->get_children();
        for (int i = 0; i <action_probs.size(); ++i) {
            move_report.children.push_back({to_string(action_probs[i].first), action_probs[i].second, children[i]->get_value(), children[i]->get_visit_count(), children[i]->get_prior()});
        }

        move_report.value = root->get_value();
        game_report.moves.push_back(move_report);

        if (board.isGameOver().second != chess::GameResult::NONE) {
            float result = 0;
            if (board.isGameOver().second != chess::GameResult::DRAW) {
                result = (board.isGameOver().second == chess::GameResult::LOSE) == 
                         (board.sideToMove() == chess::Color::BLACK) ? 1 : -1;   // if true white wins
            }

            if (result == 1) {
                Logger::log("White won");
            } else if (result == -1) {
                Logger::log("Black won");
            } else {
                Logger::log("Draw");
            }

            for (auto& item : history) {
                item.value += result;
                result = -result;
            }
            
            break;
        }
    }

    _dataset.add_data(history);
    auto game_result = board.isGameOver();
    game_report.result = game_result.second == chess::GameResult::DRAW ? "1/2 - 1/2" : (board.sideToMove() == chess::Color::WHITE ? "0-1" : "1-0");
    if (config.report_path.empty()) {
        return;
    }
    game_report.save(config.report_path + "/game_report_" + std::to_string(iteration) + "_" + std::to_string(game) + ".json");

}

void Trainer::load_model(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    _model->load(archive);

    _mcts->set_model(_model);
}

void Trainer::save_model(const std::string& path) {
    torch::serialize::OutputArchive archive;
    _model->save(archive);
    archive.save_to(path);
    Logger::log("Model saved to " + path);
}

void Trainer::set_model(std::shared_ptr<LCZero> model) {
    _model = model;
    _mcts->set_model(model);
}

void Trainer::train() {
    Logger::log("Training");
    _model->train();
    auto& trainer_config = config.training_config;
    if (!_dataset.size().has_value()) {
        Logger::log("Dataset has no data");
        return;
    }
    int size = _dataset.size().value();
    auto indices = torch::randperm(size);
    auto data_loader = torch::data::make_data_loader(
        std::move(_dataset),
        torch::data::DataLoaderOptions().batch_size(trainer_config.batch_size).workers(1)
    );

    int sum_policy_loss = 0;
    int sum_value_loss = 0;

    for (int epoch = 0; epoch < trainer_config.num_epochs; epoch++) {
        int batch_count = 0;
        for (auto& batch : *data_loader) {
            batch_count++;
            
            torch::Tensor input, policy_target, value_target;
            input = batch.input.to(torch::kCUDA);
            policy_target = batch.policy.to(torch::kCUDA);
            value_target = batch.value.view({-1}).to(torch::kCUDA);

            auto output = _model->forward(input);
            auto policy_output = std::get<0>(output).to(torch::kCUDA);
            auto value_output = std::get<1>(output).to(torch::kCUDA).view({-1});

            auto log_probs = torch::log_softmax(policy_output, 1);
            
            
            auto criterion = torch::nn::KLDivLoss(torch::nn::KLDivLossOptions(torch::kBatchMean));
            auto policy_loss = criterion(log_probs, policy_target);

            auto value_loss = torch::nn::functional::mse_loss(value_output, value_target);
            auto loss = policy_loss + value_loss;
            _model->zero_grad();
            loss.backward();
            _optimizer->step();
            sum_policy_loss += policy_loss.item<float>();
            sum_value_loss += value_loss.item<float>();
            Logger::log("Epoch: " + std::to_string(epoch) + " Batch: " + std::to_string(batch_count) + " Policy Loss: " + std::to_string(policy_loss.item<float>()) + " Value Loss: " + std::to_string(value_loss.item<float>()) + "Sum Policy Loss: " + std::to_string(sum_policy_loss) + " Sum Value Loss: " + std::to_string(sum_value_loss));
            // Logger::log("Policy Loss: " + std::to_string(policy_loss.item<float>()));
            // Logger::log("Value Loss: " + std::to_string(value_loss.item<float>()));
        }
    }
    save_model("model.pt");
    Logger::log("Model saved");
}

void Trainer::save_dataset(const std::string& path) {
    _dataset.save(path);
}

void Trainer::load_dataset(const std::string& path) {
    _dataset.load(path);
}