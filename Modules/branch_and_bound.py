# This is a Python implementation of the Branch and Bound exploration algorithm as described in the LaTeX document.
import time
import os

import numpy as np
from collections import defaultdict, Counter
# handle relative imports
if os.path.basename(os.getcwd()) != "Modules":
    from Modules.utils import SparseRepeatedArray
else:
    from utils import SparseRepeatedArray

class RECAPC:
    max_horizon = 1000
    max_seconds = 60

    def __init__(self, P, q, epsilon=1e-6, reduced_ub=False, verbose=True):
        """
        Initialize the RECAPC object.
        :param P: Probability-to-like matrix
        :param q: Initial belief state
        :param epsilon: Allowed error for the approximation
        """
        self.P = P
        self.q = q
        self.epsilon = epsilon
        self.A = list(range(len(P)))  # Assuming actions are indices of P
        self.verbose = verbose

        self.ub = np.inf
        self.n_branches = 0
        if reduced_ub:
            self.V_upper_reduced()

    def __str__(self):
        return "RECAPC"

    def c_statistic(self) -> float:
        p_min = self.P.min().item()
        p_max_neg = 1 - self.P.max().item()
        q_min = self.q.min().item()

        type_diff_min = 1
        for m1 in range(self.P.shape[1]):
            for m2 in range(m1+1, self.P.shape[1]):
                type_diff_min = min(type_diff_min, np.abs(self.P[:, m1] - self.P[:, m2]).min().item())

        top2_actions = np.partition(self.P, -2, axis=0)[-2:]
        action_diff_min = (top2_actions[1] - top2_actions[0]).min().item()

        return min(p_min, p_max_neg, q_min, type_diff_min, action_diff_min)

    def V_lower(self, q: np.ndarray) -> float:
        """
        Calculate the lower bound of the value function for belief q.
        :param P: Probability-to-like matrix
        :param q: Belief state
        :return: Lower bound of the value function
        """
        return np.max(np.sum(q.reshape(1, -1) * self.P / (1 - self.P), axis=1)).item()

    def V_upper(self, q: np.ndarray) -> float:
        """
        Calculate the upper bound of the value function for belief q.
        :param P: Probability-to-like matrix
        :param q: Belief state
        :return: Upper bound of the value function
        """
        return np.sum(q * np.max(self.P / (1 - self.P), axis=0)).item()

    def V_upper_reduced(self, n_actions=2) -> float:
        """
        Prepares a global upper bound by solving a reduced problem. Not worth it. Currently unused.
        """
        if n_actions < self.P.shape[0]:
            P_reduced = self.P.copy()
            while P_reduced.shape[0] > n_actions:
                # determine most similar rows
                min_dist = 1
                actions = (0, 1)
                for k1 in range(P_reduced.shape[0]):
                    for k2 in range(k1+1, P_reduced.shape[0]):
                        dist = np.abs(P_reduced[k1] - P_reduced[k2]).mean()
                        if dist < min_dist:
                            min_dist = dist
                            actions = (k1, k2)

                # combine the two rows
                P_reduced[actions[0]] = np.maximum(P_reduced[actions[0]], P_reduced[actions[1]])
                P_reduced = np.delete(P_reduced, actions[1], axis=0)

            # find the optimal value in the reduced problem
            bandit = RECAPC(P_reduced, self.q, reduced_ub=False, verbose=False)
            ub, _, _, nb = bandit.branch_and_bound()
            self.ub = min(self.ub, ub)
            self.n_branches += nb
        return self.ub

    @staticmethod
    def tau(P: np.ndarray, b: np.ndarray, a: int) -> np.ndarray:
        """
        Belief update function.
        :param P: Probability-to-like matrix
        :param b: Current belief state
        :param a: Action taken
        :return: Updated belief state
        """
        new_belief = b * P[a]
        new_belief /= new_belief.sum()
        return new_belief

    def branch_and_bound(self, min_horizon=0):
        """
        Perform the Branch and Bound exploration.
        :return: The approximated optimal value, the optimal prefix, and the belief transition log
        """
        best_value = self.V_lower(self.q)  # Start with the lower bound of the initial belief
        best_prefix = []
        prob_to_remain = 1
        belief = self.q
        cumulative_reward = 0
        upper_bound = self.V_upper(self.q)
        prefix_list = list()
        belief_log = [belief]
        prefix_list.append((best_value, best_prefix, prob_to_remain,
                            belief, cumulative_reward, upper_bound, belief_log))

        prefix_rewards = defaultdict(float)
        best_belief_log = belief_log

        explored_branches = self.n_branches
        start_time = time.time()

        while len(prefix_list) > 0:
            lower_bound, new_prefix, prob_to_remain, belief, cumulative_reward, upper_bound, belief_log = prefix_list.pop(
                0)

            action_counter = Counter(new_prefix)
            idx = tuple(action_counter[k] for k in range(len(self.P)))
            if prefix_rewards[idx] > cumulative_reward + 1e-8:
                continue

            explored_branches += 1
            if lower_bound >= best_value:
                best_value = lower_bound
                best_belief_log = belief_log
                best_prefix = new_prefix

            if len(new_prefix) + 1 > self.max_horizon:
                continue

            for a in self.A:
                added_belief = self.tau(self.P, belief, a)
                added_prob_to_remain = prob_to_remain * np.sum(belief * self.P[a]).item()
                added_cumulative_reward = cumulative_reward + added_prob_to_remain
                added_upper_bound = added_cumulative_reward + added_prob_to_remain * self.V_upper(added_belief)
                added_upper_bound = min(added_upper_bound, self.ub)

                new_idx = tuple(action_counter[k] + int(k == a) for k in range(len(self.P)))
                if ((added_upper_bound > best_value + self.epsilon) and
                        (prefix_rewards[new_idx] + 1e-8 < added_cumulative_reward)):
                    added_prefix = new_prefix + [a]
                    prefix_rewards[new_idx] = added_cumulative_reward

                    added_value = added_cumulative_reward + added_prob_to_remain * self.V_lower(added_belief)
                    new_belief_log = belief_log + [added_belief]
                    prefix_list.append((added_value, added_prefix, added_prob_to_remain,
                                        added_belief, added_cumulative_reward, added_upper_bound, new_belief_log))
            if time.time() - start_time > self.max_seconds:
                print('Max time exceeded!')
                break

        if self.verbose:
            print(f'explored branches: {explored_branches}')

        if len(best_prefix) > 1:
            last_action = best_prefix[-1]
            while len(best_belief_log)-1 < min_horizon:
                best_belief_log.append(self.tau(self.P, best_belief_log[-1], last_action))

        best_prefix = SparseRepeatedArray(best_prefix)
        return best_value, best_prefix, best_belief_log, explored_branches

if __name__=="__main__":
    np.random.seed(0)
    for _ in range(50):
        n_actions = 30
        n_types = 30
        P = np.random.rand(n_actions, n_types)
        q = np.random.rand(n_types)
        q /= q.sum()
        bandit = RECAPC(P, q, verbose=False)
        val, _, _, branches = bandit.branch_and_bound()
        print(f'val: {val}, branches: {branches}')