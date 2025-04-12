
#include <string>
#include <vector>
#include <torch/torch.h>
#include "node.h"
#include "chess/chess.hpp"

#ifndef STRING_UTILS_H
#define STRING_UTILS_H

inline std::string to_string(const torch::Tensor& tensor) {
    std::string res = "[";
    for (int i = 0; i < tensor.sizes().size(); ++i) {
        res += std::to_string(tensor.sizes()[i]);
        if (i < tensor.sizes().size() - 1) {
            res += ", ";
        }
    }
    res += "]";
    return res;
}

template <class T>
std::string to_string(const std::vector<T>& vec) {
    std::string res = "[";
    for (int i = 0; i < vec.size(); ++i) {
        res += std::to_string(vec[i]);
        if (i < vec.size() - 1) {
            res += ", ";
        }
    }
    res += "]";
    return res;
}

inline std::string to_string(const char* str) {
    return std::string(str);
}

inline std::string to_string(const int& x) {
    return std::to_string(x);
}

inline std::string to_string(const float& x) {
    return std::to_string(x);
}

inline std::string to_string(const double& x) {
    return std::to_string(x);
}

inline std::string to_string(const std::string& str) {
    return str;
}

inline std::string to_string(const torch::IntArrayRef& arr) {
    std::string res = "[";
    for (int i = 0; i < arr.size(); ++i) {
        res += std::to_string(arr[i]);
        if (i < arr.size() - 1) {
            res += ", ";
        }
    }
    res += "]";
    return res;
}

template <typename Head, typename... Tail>
std::string join_str(const std::string& sep, Head&& head, Tail&&... tail) {
    std::string res = "";
    res += to_string(head);
    if constexpr (sizeof...(tail) > 0) {
        res += sep + join_str(sep, std::forward<Tail>(tail)...);
    }
    return res;
}


inline std::string to_string(const node_t& node, int tab_count = 0) {
    std::string res = join_str
        (" ", std::string(tab_count, '\t'), "Node", node.move.from(), node.move.to(), "Value", node.value, "Visit Count", node.visit_count);
    for (const auto& child : node.children) {
        res += "\n";
        for (int i = 0; i < tab_count + 1; ++i) {
            res += "\t";
        }
        res += to_string(*child, tab_count + 1);
    }
    return res;
}

inline std::string to_string(const chess::Move& move) {
    return std::string(move.from()) + " " + std::string(move.to());
}

// Split like python
inline std::vector<std::string> split(std::string str, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        tokens.push_back(str.substr(0, pos));
        str.erase(0, pos + delimiter.length());
    }
    tokens.push_back(str);
    return tokens;
}
#endif // STRING_UTILS_H