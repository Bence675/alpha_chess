
#include <torch/torch.h>
#include <memory>
#include "model.h"
#include "mcts.h"

struct TrainerConfig {
    int num_simulations = 1000;
    int num_games = 100;
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
    void set_model(std::shared_ptr<Model> model);

private:
    std::shared_ptr<Model> _model;
    std::shared_ptr<MCTS> _mcts;
    TrainerConfig _config;
};