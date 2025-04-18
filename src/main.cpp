
#include <iostream>
#include <memory>

#include "logger/logger.h"
#include "trainer/trainer.h"
#include "args/args_parser.h"

int main(int argc, char *argv[]) {

    auto args = args::parse_args(argc, argv);
    if (args.help) {
        args::print_help();
        return 0;
    }

    auto config = config::load_config(args.config_file);

    Trainer trainer(config);
    for (int i = 1; i <= 100; i++) {
        Logger::log("Iteration " + std::to_string(i));
        memory::clear();
        trainer.self_play(i);
        trainer.save_dataset("dataset1");
        
        trainer.load_dataset("dataset1");
        trainer.self_play(i);
        trainer.train();
    }
    // trainer.train();


    return 0;
}