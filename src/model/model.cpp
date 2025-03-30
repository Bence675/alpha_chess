
#include <memory>
#include <vector>
#include "model.h"
#include <logger.h>


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
}

ConvModel::ConvModel(int64_t num_actions)
    : conv_block(20, 16, 3, 1, 1),
        policy_head(16, 2, 1, 1, 0, num_actions),
        value_head(16, 1, 1, 1, 0, 16) {}

std::tuple<torch::Tensor, torch::Tensor> ConvModel::forward(torch::Tensor x) {
    Logger::log("Forward pass");
    x = conv_block.forward(x);
    torch::Tensor policy = policy_head.forward(x);
    torch::Tensor value = value_head.forward(x);
    return std::make_tuple(policy, value);
}