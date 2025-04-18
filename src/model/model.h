

#include <torch/torch.h>
#include <torch/serialize.h>
#include <vector>
#include "logger.h"
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include "config.h"


#ifndef MODEL_H
#define MODEL_H

class SEBlock : public torch::nn::Module {
    public:
        SEBlock(int64_t in_channels, int64_t reduction_ratio = 16) {
            _sequential = torch::nn::Sequential(
                torch::nn::AdaptiveAvgPool2d(1),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels / reduction_ratio, 1)),
                torch::nn::ReLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels / reduction_ratio, in_channels, 1)),
                torch::nn::Sigmoid()
            );
            register_module("sequential", _sequential);
        }

        torch::Tensor forward(torch::Tensor x) {
            auto se = _sequential->forward(x);
            return x * se;
        }

        void to(torch::Device device) {
            _sequential->to(device);
        }
    private:
        torch::nn::Sequential _sequential;
};

class ResBlock : public torch::nn::Module {
    public:
        ResBlock(int64_t in_channels, int64_t out_channels, int64_t reduction_ratio = 16) {
            int64_t kernel_size = 3;
            int64_t stride = 1;
            int64_t padding = 1;
            _conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding));
            _bn1 = torch::nn::BatchNorm2d(out_channels);
            _relu = torch::nn::ReLU();
            _conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(padding));
            _bn2 = torch::nn::BatchNorm2d(out_channels);
            _se_block = torch::nn::ModuleHolder<SEBlock>(out_channels, reduction_ratio);
            register_module("conv1", _conv1);
            register_module("bn1", _bn1);
            register_module("relu", _relu);
            register_module("conv2", _conv2);
            register_module("bn2", _bn2);
            register_module("se_block", _se_block);
        }

        torch::Tensor forward(torch::Tensor x) {
            auto identity = x;
            x = _conv1->forward(x);
            x = _bn1->forward(x);
            x = _relu->forward(x);
            x = _conv2->forward(x);
            x = _bn2->forward(x);
            x = _se_block->forward(x);
            return x + identity;
        }

        void to(torch::Device device) {
            _conv1->to(device);
            _bn1->to(device);
            _relu->to(device);
            _conv2->to(device);
            _bn2->to(device);
            _se_block->to(device);
        }
    private:
        torch::nn::Conv2d _conv1 = nullptr;
        torch::nn::BatchNorm2d _bn1 = nullptr;
        torch::nn::ReLU _relu;
        torch::nn::Conv2d _conv2 = nullptr;
        torch::nn::BatchNorm2d _bn2 = nullptr;
        torch::nn::ModuleHolder<SEBlock> _se_block = nullptr;
};

class ResNet : public torch::nn::Module {
    public:
        ResNet(int64_t in_channels, int64_t out_channels, int64_t num_blocks, int64_t reduction_ratio = 16) {
            _residual_blocks = torch::nn::Sequential();
            for (int i = 0; i < num_blocks; i++) {
                _residual_blocks->push_back(ResBlock(out_channels, out_channels, reduction_ratio));
            }
            register_module("residual_blocks", _residual_blocks);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = _residual_blocks->forward(x);
            return x;
        }

        void to(torch::Device device) {
            _residual_blocks->to(device);
        }
    private:
        torch::nn::Sequential _residual_blocks;
};

class PolicyHead : public torch::nn::Module {
    public:
        PolicyHead(int64_t in_channels) {
            _conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1));
            _conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 73, 3).padding(1));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = _conv1->forward(x);
            x = _conv2->forward(x);
            return x.view({x.size(0), -1});
        }

        void to(torch::Device device) {
            _conv1->to(device);
            _conv2->to(device);
        }
    private:
        torch::nn::Conv2d _conv1 = nullptr;
        torch::nn::Conv2d _conv2 = nullptr;
};

class ValueHead : public torch::nn::Module {
    public:
        ValueHead(int64_t in_channels) {
            _conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 32, 3).padding(1));
            _conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 2, 1));
            _relu = torch::nn::ReLU();
            _fc = torch::nn::Linear(128, 1);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = _conv1->forward(x);
            x = _conv2->forward(x);
            x = _relu->forward(x);
            x = x.view({x.size(0), -1});
            x = _fc->forward(x);
            return x;
        }

        void to(torch::Device device) {
            _conv1->to(device);
            _conv2->to(device);
            _fc->to(device);
        }
    private:
        torch::nn::Conv2d _conv1 = nullptr;
        torch::nn::Conv2d _conv2 = nullptr;
        torch::nn::ReLU _relu;
        torch::nn::Linear _fc = nullptr;
};

class LCZero : public torch::nn::Module {
    public:
        LCZero(const config::Config::NetworkConfig& config) {
            _conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(config.in_channels, config.num_filters, 3).padding(1));
            _resnet = torch::nn::ModuleHolder<ResNet>(config.num_filters, config.num_filters, config.num_blocks, config.reduction_ratio);
            _relu = torch::nn::ReLU();
            _policy_head = torch::nn::ModuleHolder<PolicyHead>(config.num_filters);
            _value_head = torch::nn::ModuleHolder<ValueHead>(config.num_filters);
            register_module("conv", _conv);
            register_module("resnet", _resnet);
            register_module("policy_head", _policy_head);
            register_module("value_head", _value_head);
        }

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
            if (x.device() != torch::kCUDA) {
                x = x.to(torch::kCUDA);
            }
            x = _conv->forward(x);
            x = _resnet->forward(x);
            x = _relu->forward(x);
            auto policy = _policy_head->forward(x);
            auto value = _value_head->forward(x);

            /*policy = torch::nn::functional::softmax(
                policy,
                torch::nn::functional::SoftmaxFuncOptions(1).dtype(torch::kFloat32)
            );*/
            value = torch::tanh(value);
            return std::make_tuple(policy, value);
        }

        void to(torch::Device device) {
            _conv->to(device);
            _resnet->to(device);
            _policy_head->to(device);
            _value_head->to(device);
        }

    private:
        torch::nn::Conv2d _conv = nullptr;
        torch::nn::ModuleHolder<ResNet> _resnet = nullptr;
        torch::nn::ReLU _relu;
        torch::nn::ModuleHolder<PolicyHead> _policy_head = nullptr;
        torch::nn::ModuleHolder<ValueHead> _value_head = nullptr;
};

/*
class ConvModel : public torch::nn::Module  {
public:
    ConvModel(const config::Config::NetworkConfig& config);
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    void to(torch::Device device) {
        _conv_block->to(device);
        _policy_head->to(device);
        _value_head->to(device);
    }

private:
    
    torch::nn::Sequential _conv_block;
    torch::nn::Sequential _policy_head;
    torch::nn::Sequential _value_head;

};*/
#endif // MODEL_H