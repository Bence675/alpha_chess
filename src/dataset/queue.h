
#include <vector>
#include <mutex>
#include <stdexcept>

template<typename T>
class Queue {
public:
    Queue(int max_size=0) : _max_size(max_size) {}

    T operator[](size_t index) const {
        std::lock_guard<std::mutex> lock(_mtx);
        if (index >= _queue.size()) {
            throw std::runtime_error("Index out of range");
        }
        return _queue[index];
    }

    void push(const T& data) {
        std::lock_guard<std::mutex> lock(_mtx);
        if (_max_size > 0 && _queue.size() >= _max_size * 2) {
            _queue.erase(_queue.begin(), _queue.begin() + _max_size);
        }
        _queue.push_back(data);
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(_mtx);
        return _queue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(_mtx);
        if (_max_size == 0) {
            return _queue.size();
        }
        return std::min(_queue.size(), static_cast<size_t>(_max_size));
    }

    void clear(){
        std::lock_guard<std::mutex> lock(_mtx);
        _queue.clear();
    }

    Queue(const Queue&) = default;
    Queue(Queue&&) = default;
    Queue& operator=(Queue&&) = default;
    ~Queue() = default;
    Queue& operator=(const Queue& other) {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(_mtx);
            _queue = other._queue;
            _max_size = other._max_size;
        }
        return *this;
    }


private:
    mutable std::mutex _mtx;
    std::vector<T> _queue;
    int _max_size;
};