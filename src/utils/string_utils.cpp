
#include "string_utils.h"


using namespace std;

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

template <class Head, class ...Tail>
[[nodiscard]] std::string join_str(const std::string& sep, Head&& head, Tail&&... tail) {
    return to_string(std::forward<Head>(head)) + (... + (sep + to_string(std::forward<Tail>(tail))));
}