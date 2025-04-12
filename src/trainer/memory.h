


class memory
{
public:
    std::unordered_map<std::string, std::pair<node_t::action_probs_t, int>> action_probs_map{};
    std::vector<std::string> boards_to_compute{};
    std::vector<std::string> processing{};
    std::mutex boards_to_compute_and_processing_mutex;
    std::mutex action_probs_map_mutex;

    std::vector<std::pair<std::string, int>> self_play_memory{};
    std::mutex self_play_memory_mutex;
    static memory& getInstance() {
        static memory instance;
        return instance;
    }
    memory(const memory&) = delete;
    memory& operator=(const memory&) = delete;
    memory(memory&&) = delete;
    memory& operator=(memory&&) = delete;
    memory() = default;
    ~memory() = default;
};