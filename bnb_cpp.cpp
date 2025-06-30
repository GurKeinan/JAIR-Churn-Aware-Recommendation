#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono> // For timing
#include <future>
#include <mutex>
#include <tuple>
#include <queue>
#include <utility>  // for std::index_sequence
#include <fstream>
#include "json.hpp"  // Include the JSON library

using json = nlohmann::json;

using namespace std;

// Hash function for std::vector<int>
struct vector_hash {
    size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for(auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Function to convert a vector to a tuple dynamically
template<typename T>
std::tuple<> vector_to_tuple(const std::vector<T>& vec) {
    std::tuple<> result;
    for (size_t i = 0; i < vec.size(); ++i) {
        result = std::tuple_cat(result, std::make_tuple(vec[i]));
    }
    return result;
}

class DepartingBandits {
public:
    DepartingBandits(
            const vector<vector<double>>& P,
            const vector<double>& q,
            double epsilon
        )
        : P(P), q(q), epsilon(epsilon) {
        for (int i = 0; i < P.size(); ++i) {
            A.push_back(i);
        }
        precompute_values(); // Precompute frequently used values
    }

    double V_lower(const vector<double>& q) {
        double max_val = -1;
        for (int a = 0; a < P.size(); ++a) {
            double sum_val = 0;
            for (int s = 0; s < q.size(); ++s) {
                sum_val += q[s] * precomputed_values[a][s];
            }
            max_val = max(max_val, sum_val);
        }
        return max_val;
    }

    double V_upper(const vector<double>& q) {
        double sum_val = 0;
        for (int s = 0; s < q.size(); ++s) {
            double max_val = -1;
            for (int a = 0; a < P.size(); ++a) {
                max_val = max(max_val, precomputed_values[a][s]);
            }
            sum_val += q[s] * max_val;
        }
        return sum_val;
    }

    vector<double> tau(const vector<double>& b, int a) {
        vector<double> new_belief(b.size(), 0);
        double temp = 0;
        for (int s = 0; s < b.size(); ++s) {
            new_belief[s] = b[s] * P[a][s];
            temp += new_belief[s];
        }
        if (temp == 0) temp = 1; // Prevent division by zero
        for (int s = 0; s < new_belief.size(); ++s) {
            new_belief[s] /= temp;
        }
        return new_belief;
    }

    tuple<double, vector<int>, int> branch_and_bound() {
        double best_value = V_lower(q);
        vector<int> best_prefix;
        double prob_to_remain = 1;
        vector<double> belief = q;
        double cumulative_reward = 0;
        double upper_bound = V_upper(q);
        int n_branches = 0;

        queue<tuple<double, vector<int>, double, vector<double>, double, double>> prefix_list;
        // priority_queue<
        // tuple<double, vector<int>, double, vector<double>, double, double>,
        // vector<tuple<double, vector<int>, double, vector<double>, double, double>>,
        // std::greater<>
        // > prefix_list;

        prefix_list.emplace(best_value, best_prefix, prob_to_remain, belief,
            cumulative_reward, upper_bound);

        // unordered_map<gen_tuple, double, gen_tuple_hash> prefix_rewards;
        std::unordered_map<std::vector<int>, double, vector_hash> prefix_rewards;

        mutex mtx;  // Mutex to protect shared variables
        vector<future<void>> futures;  // To hold futures from async tasks

        // Pop several nodes from the stack
        const int batch_size = 32;

        while (!prefix_list.empty()) {
            vector<tuple<double, vector<int>, double, vector<double>, double, double>> nodes_to_process;

            for (int i = 0; i < batch_size && !prefix_list.empty(); ++i) {
                nodes_to_process.push_back(prefix_list.front());
                prefix_list.pop();
            }

            // Process the batch of nodes in parallel
            for (auto& node : nodes_to_process) {
                futures.push_back(async(launch::async, [&]() {
                    auto [lower_bound, new_prefix, prob_to_remain, belief,
                          cumulative_reward, upper_bound] = node;

                    // Create a tuple index from the action counts
                    vector<int> action_counts(P.size(), 0);
                    for (int action : new_prefix) {
                        action_counts[action]++;
                    }

                    // Check the reward condition
                    if (prefix_rewards[action_counts] > cumulative_reward + epsilon) {
                        return; // Skip processing this node
                    }

                    {
                        lock_guard<mutex> lock(mtx);
                        if (lower_bound > best_value || (lower_bound >= best_value && new_prefix.size() > best_prefix.size())) {
                            best_value = lower_bound;
                            best_prefix = new_prefix;
                        }
                        n_branches++;
                    }


                    if (lower_bound < upper_bound - epsilon) {
                        vector<tuple<double, vector<int>, double, vector<double>, double, double>> new_nodes;

                        for (int a : A) {
                            vector<int> added_prefix = new_prefix;
                            added_prefix.push_back(a);
                            vector<double> added_belief = tau(belief, a);

                            double prob_mult = 0;
                            for (int s = 0; s < belief.size(); ++s) {
                                prob_mult += belief[s] * P[a][s];
                            }
                            double added_prob_to_remain = prob_to_remain * prob_mult;

                            double added_cumulative_reward = cumulative_reward + added_prob_to_remain;
                            double added_upper_bound = added_cumulative_reward + added_prob_to_remain * V_upper(added_belief);

                            if (added_upper_bound < best_value + epsilon) {
                                continue;
                            }

                            double added_value = added_cumulative_reward + added_prob_to_remain * V_lower(added_belief);

                            action_counts[a]++;
                            {
                                lock_guard<mutex> lock(mtx);
                                if (prefix_rewards[action_counts] > added_cumulative_reward) {
                                    action_counts[a]--;
                                    continue;
                                }
                                prefix_rewards[action_counts] = added_cumulative_reward;
                            }
                            action_counts[a]--;
                            new_nodes.push_back(make_tuple(added_value, added_prefix, added_prob_to_remain,
                                added_belief, added_cumulative_reward, added_upper_bound));
                        }

                        // Add new nodes to the prefix_list
                        lock_guard<mutex> lock(mtx);
                        for (auto& new_node : new_nodes) {
                            prefix_list.push(new_node);
                        }
                    }
                }));
            }

            // Wait for all tasks in the batch to complete
            for (auto& future : futures) {
                future.get();
            }
            futures.clear();
        }

        return make_tuple(best_value, best_prefix, n_branches);
    }

private:
    vector<vector<double>> P;
    vector<double> q;
    double epsilon;
    vector<int> A;
    vector<vector<double>> precomputed_values; // Precomputed values

    void precompute_values() {
        precomputed_values.resize(P.size(), vector<double>(q.size()));
        for (int a = 0; a < P.size(); ++a) {
            for (int s = 0; s < q.size(); ++s) {
                precomputed_values[a][s] = P[a][s] / max(1e-9, (1 - P[a][s]));
            }
        }
    }
};

tuple<double, vector<int>, int, double> run_bnb(
    vector<vector<double>> P,
    vector<double> q,
    double epsilon = 1e-6)
{
    DepartingBandits bb(P, q, epsilon);

    auto start = chrono::high_resolution_clock::now();

    auto [value, best_prefix, n_branches] = bb.branch_and_bound();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    return make_tuple(value, best_prefix, n_branches, duration.count());
}

int main(int argc, char* argv[]) {
    double epsilon = 1e-8;

    std::string input;
    std::getline(std::cin, input);

    // Read P and q from the file
    json j = json::parse(input);
    std::vector<std::vector<double>> P = j["P"];
    std::vector<double> q = j["q"];

    if (argc > 1) {
        epsilon = stod(argv[1]);
    }
    auto [value, best_prefix, n_branches, duration] = run_bnb(P, q, epsilon);

    // Output results
    cout << "Optimal value: " << value << endl;
    cout << "Optimal prefix: [";
    for (int a : best_prefix) {
        cout << a << ", ";
    }
    cout << "]" << endl;
    cout << "Number of branches: " << n_branches << endl;

    // Output the execution time
    cout << "Execution time (seconds): " << duration << endl;
    return 0;
}
