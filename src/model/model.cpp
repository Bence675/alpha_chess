
#include <memory>
#include <vector>
#include "model.h"
#include <logger.h>
#include "string_utils.h"




torch::nn::Sequential create_conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding) {
    return std::move(torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    ));
}

torch::nn::Sequential create_policy_head(int64_t in_dimensions, int64_t out_channels, int64_t hidden_dim) {
    return std::move(torch::nn::Sequential(
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(in_dimensions, hidden_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(hidden_dim, out_channels),
        torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1))
    ));
}

torch::nn::Sequential create_value_head(int64_t in_dimensions, int64_t out_channels, int64_t hidden_dim) {
    return std::move(torch::nn::Sequential(
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(in_dimensions, hidden_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(hidden_dim, out_channels),
        torch::nn::Tanh()
    ));
}

/*
conv_block_t::conv_block_t(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding)
    : conv1(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        bn1(out_channels),
        relu1(torch::nn::ReLUOptions().inplace(true)),
        conv2(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        bn2(out_channels),
        relu2(torch::nn::ReLUOptions().inplace(true))
{
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("relu1", relu1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("relu2", relu2);
}

torch::Tensor conv_block_t::forward(torch::Tensor x)
{
    x = conv1(x);
    x = bn1(x);
    x = relu1(x);
    x = conv2(x);
    x = bn2(x);
    x = relu2(x);
    return x;
}


policy_head_t::policy_head_t(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t num_actions)
    : conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        bn(out_channels),
        relu(torch::nn::ReLUOptions().inplace(true)),
        fc(torch::nn::LinearOptions(out_channels * 8 * 8, num_actions))
{
    register_module("conv", conv);
    register_module("bn", bn);
    register_module("relu", relu);
    register_module("fc", fc);
}

torch::Tensor policy_head_t::forward(torch::Tensor x)
{
    x = conv(x);
    x = bn(x);
    x = relu(x);
    x = x.view({x.size(0), -1});
    x = fc(x);
    return x;
}

value_head_t::value_head_t(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t hidden_dim)
    : conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding)),
    bn(out_channels),
    relu(torch::nn::ReLUOptions().inplace(true)),
    fc1(torch::nn::LinearOptions(out_channels * 8 * 8, hidden_dim)),
    relu1(torch::nn::ReLUOptions().inplace(true)),
    fc2(torch::nn::LinearOptions(hidden_dim, 1)),
    tanh(torch::nn::Tanh())
{
    register_module("conv", conv);
    register_module("bn", bn);
    register_module("relu", relu);
    register_module("fc1", fc1);
    register_module("relu1", relu1);
    register_module("fc2", fc2);
    register_module("tanh", tanh);
}

torch::Tensor value_head_t::forward(torch::Tensor x)
{
    x = conv(x);
    x = bn(x);
    x = relu(x);
    x = x.view({x.size(0), -1});
    x = fc1(x);
    x = relu1(x);
    x = fc2(x);
    x = tanh(x);
    return x;
}*/

ConvModel::ConvModel (int num_actions, int in_channels, int hidden_channels, int out_channels, int kernel_size, int hidden_dim)
{
    _conv_block = create_conv2d(in_channels, hidden_channels, kernel_size, 1, 1);
    _policy_head = create_policy_head(hidden_channels * 8 * 8, 64 * 73, hidden_dim);
    _value_head = create_value_head(hidden_channels * 8 * 8, 1, hidden_dim);

    register_module("conv_block", _conv_block);
    register_module("policy_head", _policy_head);
    register_module("value_head", _value_head);
}

std::tuple<torch::Tensor, torch::Tensor> ConvModel::forward(torch::Tensor x) {
    if (x.device() != torch::kCUDA) {
        Logger::log("Input tensor is not on CUDA device");
        x = x.to(torch::kCUDA);
    }
    x = _conv_block->forward(x);
    x = x.view({x.size(0), -1});
    Logger::log("Shape after conv block: " + to_string(x.sizes()));
    torch::Tensor policy = _policy_head->forward(x);
    Logger::log("Shape after policy head: " + to_string(policy.sizes()));
    torch::Tensor value = _value_head->forward(x);
    Logger::log("Shape after value head: " + to_string(value.sizes()));
    return std::make_tuple(policy, value);
}

// void Convtorch::nn::Module::eval() {
//     conv_block.eval();
//     policy_head.eval();
//     value_head.eval();   
// }
// 
// void Convtorch::nn::Module::train() {
//     conv_block.train();
//     policy_head.train();
//     value_head.train();
// }
// 
// 