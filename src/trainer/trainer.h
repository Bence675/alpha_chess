
#include <torch/torch.h>
#include <memory>
#include "model.h"
#include "mcts.h"
#include "memory.h"
#include "dataset.h"

#ifndef TRAINER_H
#define TRAINER_H

struct TrainerConfig {
    int num_simulations = 1000;
    int num_games = 100;
    int max_threads = 128;
    int num_epochs = 10;
    float c_puct = 1.0;
    float epsilon = 0.03;
};


class Trainer {
public:
    Trainer(TrainerConfig config = TrainerConfig());

    void train();
    void play_game();
    void self_play();
    void load_model(const std::string& path);
    void save_model(const std::string& path);
    void set_model(std::shared_ptr<ConvModel> model);


private:
    std::shared_ptr<ConvModel> _model;
    std::shared_ptr<MCTS> _mcts;
    std::thread model_thread;
    TrainerConfig config;
    ChessDataSet _dataset;
    std::shared_ptr<torch::optim::Adam> _optimizer;
};

#endif // TRAINER_H