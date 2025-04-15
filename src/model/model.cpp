
#include <memory>
#include <vector>
#include "model.h"
#include <logger.h>
#include "string_utils.h"




torch::nn::Sequential create_conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding) {
    return std::move(torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions()),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions())
    ));
}

torch::nn::Sequential create_policy_head(int64_t in_dimensions, int64_t out_channels, int64_t hidden_dim) {
    return std::move(torch::nn::Sequential(
        torch::nn::ReLU(torch::nn::ReLUOptions()),
        torch::nn::Linear(in_dimensions, hidden_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions()),
        torch::nn::Linear(hidden_dim, out_channels),
        torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1))
    ));
}

torch::nn::Sequential create_value_head(int64_t in_dimensions, int64_t out_channels, int64_t hidden_dim) {
    return std::move(torch::nn::Sequential(
        torch::nn::ReLU(torch::nn::ReLUOptions()),
        torch::nn::Linear(in_dimensions, hidden_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions()),
        torch::nn::Linear(hidden_dim, out_channels),
        torch::nn::Tanh()
    ));
}


ConvModel::ConvModel (const config::Config::NetworkConfig& config) {
    int64_t hidden_channels = config.num_hidden_channels;
    int64_t hidden_dim = config.num_hidden_dimensions;
    
    _conv_block = create_conv2d(19, hidden_channels, 3, 1, 1);
    _policy_head = create_policy_head(hidden_channels * 8 * 8, 64 * 73, hidden_dim);
    _value_head = create_value_head(hidden_channels * 8 * 8, 1, hidden_dim);

    register_module("conv_block", _conv_block);
    register_module("policy_head", _policy_head);
    register_module("value_head", _value_head);
}

std::tuple<torch::Tensor, torch::Tensor> ConvModel::forward(torch::Tensor x) {
    if (x.device() != torch::kCUDA) {
        x = x.to(torch::kCUDA);
    }
    x = _conv_block->forward(x);
    x = x.view({x.size(0), -1});
    // Logger::log("Shape after conv block: " + to_string(x.sizes()));
    torch::Tensor policy = _policy_head->forward(x);
    // Logger::log("Shape after policy head: " + to_string(policy.sizes()));
    torch::Tensor value = _value_head->forward(x);
    //Logger::log("Shape after value head: " + to_string(value.sizes()));
    return std::make_tuple(policy, value);
}
