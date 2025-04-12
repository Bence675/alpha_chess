
#include <queue>
#include <mutex>
#include <stdexcept>

template<typename T>
class Queue {
public:
    Queue(int max_size=0) : _max_size(max_size) {}

    void push(const T& data) {
        std::lock_guard<std::mutex> lock(_mtx);
        if (_max_size > 0 && _queue.size() >= _max_size) {
            _queue.pop();
        }
        _queue.push(data);
    }
    

    T pop() {
        std::lock_guard<std::mutex> lock(_mtx);
        if (_queue.empty()) {
            throw std::runtime_error("Queue is empty");
        }
        T data = _queue.front();
        _queue.pop();
        return data;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(_mtx);
        return _queue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(_mtx);
        return _queue.size();
    }
    void clear(){
        std::lock_guard<std::mutex> lock(_mtx);
        while (!_queue.empty()) {
            _queue.pop();
        }
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
    std::queue<T> _queue;
    int _max_size;
};