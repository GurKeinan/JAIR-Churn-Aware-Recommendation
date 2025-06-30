# This is a Python implementation of the Branch and Bound exploration algorithm as described in the LaTeX document.
import copy
import time

import numpy as np
import matplotlib.pyplot as plt


def drop_controlled_actions(prob_mat):
    """
    Drop actions that are controlled by other actions.
    """
    controlled_actions = []
    for action1 in range(len(prob_mat)):
        for action2 in range(len(prob_mat)):
            if action1 != action2:
                if np.all(prob_mat[action1] >= prob_mat[action2]):
                    controlled_actions.append(action2)

    controlled_actions = list(set(controlled_actions))
    prob_mat_changed = np.delete(prob_mat, controlled_actions, axis=0)
    return prob_mat_changed


class RECAPC:
    def __init__(self, P, q, epsilon):
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

    def V_lower(self, q: np.ndarray) -> float:
        """
        Calculate the lower bound of the value function for belief q.
        :param q: Belief state
        :return: Lower bound of the value function
        """
        return np.max([sum(q[s] * P_a_s / (1 - P_a_s) for s, P_a_s in enumerate(P_a)) for P_a in self.P])

    def V_upper(self, q: np.ndarray) -> float:
        """
        Calculate the upper bound of the value function for belief q.
        :param q: Belief state
        :return: Upper bound of the value function
        """
        return sum(q[s] * max(P_a_s / (1 - P_a_s) for P_a_s in P_s) for s, P_s in enumerate(zip(*self.P)))

    def tau(self, b: np.ndarray, a: int) -> np.ndarray:
        """
        Belief update function.
        :param b: Current belief state
        :param a: Action taken
        :return: Updated belief state
        """
        new_belief = np.zeros(len(b))
        temp = 0
        for s in range(len(b)):
            new_belief[s] = b[s] * self.P[a][s]
            temp += new_belief[s]
        new_belief /= temp
        return new_belief

    def branch_and_bound(self):
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
        prefix_list.append((best_value, best_prefix, prob_to_remain, belief, cumulative_reward, upper_bound))

        belief_transition_log = dict()

        while len(prefix_list) > 0:
            lower_bound, new_prefix, prob_to_remain, belief, cumulative_reward, upper_bound = prefix_list.pop(0)

            belief_transition_log[tuple(belief)] = []

            if lower_bound > best_value:
                best_value = lower_bound
                best_prefix = new_prefix

            if lower_bound >= upper_bound - self.epsilon:
                continue

            else:

                for a in self.A:

                    added_prefix = new_prefix + [a]
                    added_belief = self.tau(belief, a)
                    belief_transition_log[tuple(added_belief)] = []
                    added_prob_to_remain = prob_to_remain * \
                        np.sum([belief[s] * self.P[a][s] for s in range(len(belief))])
                    added_cumulative_reward = cumulative_reward + added_prob_to_remain
                    added_upper_bound = added_cumulative_reward + added_prob_to_remain * self.V_upper(added_belief)

                    belief_transition_log[tuple(belief)] += [tuple(added_belief)]

                    if added_upper_bound < best_value + self.epsilon:
                        continue

                    added_value = added_cumulative_reward + added_prob_to_remain * self.V_lower(added_belief)
                    prefix_list.append((added_value, added_prefix, added_prob_to_remain,
                                       added_belief, added_cumulative_reward, added_upper_bound))

        return best_value, best_prefix, belief_transition_log


TITLE_SIZE = 20
AXIS_SIZE = 16
TEXT_SIZE = 14
LEGEND_SIZE = 16
INITIAL_BELIEF_SIZE = 16
ARROW_LENGTH = 0.03
ARROW_WIDTH = 0.02
PLOT_TITLE = False


def plot_policies(P, q, epsilon, num_of_steps_for_optimal, num_of_steps_for_greedy, num_of_steps_for_single, list_of_texts, name_to_save, plot_legend=True):
    bb = RECAPC(P, q, epsilon)
    value, best_prefix, belief_log = bb.branch_and_bound()

    if len(best_prefix) < num_of_steps_for_optimal:
        current_belief = copy.deepcopy(q)
        for action in best_prefix:
            current_belief = bb.tau(current_belief, action)
        optimal_action_after_prefix = np.argmax(
            [sum(current_belief[s] * P_a_s / (1 - P_a_s) for s, P_a_s in enumerate(P_a)) for P_a in P])
        best_prefix += [optimal_action_after_prefix] * (num_of_steps_for_optimal - len(best_prefix))

    # Set the desired DPI (dots per inch) for high resolution
    dpi_value = 300  # You can adjust this value as needed

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi_value)

    # Set limits and labels
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    if PLOT_TITLE:
        ax.set_title('Belief Walk in 2D Simplex', fontsize=TITLE_SIZE)
    ax.set_xlabel(r'$\mathbf{b}(m_1)$', fontsize=AXIS_SIZE)
    ax.set_ylabel(r'$\mathbf{b}(m_2)$', fontsize=AXIS_SIZE)

    # Draw the simplex (triangle) with vertices (0, 0), (1, 0), (0, 1)
    simplex_vertices = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    ax.plot(simplex_vertices[:, 0], simplex_vertices[:, 1], 'k-')

    # Colors for each loop
    colors = ['#4682B4', '#3CB371', '#F08080']

    # Plot best_prefix and annotate with numbers
    current_belief = copy.deepcopy(q)
    ax.plot(current_belief[0], current_belief[1], marker='o', color='black', markersize=INITIAL_BELIEF_SIZE)
    for i, action in enumerate(best_prefix):
        next_belief = bb.tau(current_belief, action)
        ax.arrow(current_belief[0], current_belief[1], next_belief[0] - current_belief[0],
                 next_belief[1] - current_belief[1],
                 head_width=ARROW_WIDTH, head_length=ARROW_LENGTH, fc=colors[0], ec=colors[0])
        # Annotate with the belief number in bold
        if i + 1 in list_of_texts:
            ax.text(next_belief[0], next_belief[1], str(i + 1), fontsize=TEXT_SIZE, color='black', weight='bold')
        current_belief = next_belief
    ax.plot([], [], 'o', color=colors[0], label='Optimal Policy')

    # Plot the best single action
    best_single_action = [np.argmax([sum(q[s] * P_a_s / (1 - P_a_s)
                                    for s, P_a_s in enumerate(P_a)) for P_a in P])] * num_of_steps_for_single
    current_belief = copy.deepcopy(q)
    # ax.plot(current_belief[0], current_belief[1], 'bo')
    for action in best_single_action:
        next_belief = bb.tau(current_belief, action)
        ax.arrow(current_belief[0], current_belief[1], next_belief[0] - current_belief[0],
                 next_belief[1] - current_belief[1],
                 head_width=ARROW_WIDTH, head_length=ARROW_LENGTH, fc=colors[1], ec=colors[1])
        current_belief = next_belief
    ax.plot([], [], 'o', color=colors[1], label='Best Fixed-Action Policy')

    # Plot the greedy action
    greedy_prefix = []
    greedy_belief = copy.deepcopy(q)
    for _ in range(num_of_steps_for_greedy):
        greedy_action = np.argmax([sum(greedy_belief[s] * P_a_s for s, P_a_s in enumerate(P_a)) for P_a in P])
        greedy_prefix.append(greedy_action)
        greedy_belief = bb.tau(greedy_belief, greedy_action)

    current_belief = copy.deepcopy(q)
    # ax.plot(current_belief[0], current_belief[1], 'go')
    for action in greedy_prefix:
        next_belief = bb.tau(current_belief, action)
        ax.arrow(current_belief[0], current_belief[1], next_belief[0] - current_belief[0],
                 next_belief[1] - current_belief[1],
                 head_width=ARROW_WIDTH, head_length=ARROW_LENGTH, fc=colors[2], ec=colors[2])
        current_belief = next_belief
    ax.plot([], [], 'o', color=colors[2], label='Myopic Policy')

    # Add legend
    if plot_legend:
        ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.show()

    fig.savefig(name_to_save, dpi=dpi_value)


EPSILON = 1e-6


if __name__ == '__main__':
    # Initialize your variables and objects here...

    # Example matrices and beliefs
    P1 = np.array([
        [0.8611176, 0.45909642, 0.68624171],
        [0.09692695, 0.55311651, 0.86037899],
        [0.50548523, 0.14297444, 0.88789585]
    ])
    q1 = np.array([0.17126332, 0.44649995, 0.38223673])

    P2 = np.array([
        [0.68476253, 0.90999066, 0.54565803],
        [0.77407938, 0.82840092, 0.5833229],
        [0.19305288, 0.91269007, 0.52734199]
    ])

    q2 = np.array([0.38441013, 0.11967344, 0.49591643])

    P3 = np.array([
        [0.54920378, 0.05598237, 0.88778838],
        [0.21953968, 0.85762727, 0.20719559],
        [0.76742632, 0.79915476, 0.40508675]
    ])

    q3 = np.array([0.29719289, 0.4000606, 0.30274651])

    P4 = np.array([[0.40109148, 0.85207045, 0.83007302], [0.76832171, 0.78372429, 0.83142133]])
    q4 = np.array([0.37554162, 0.392054, 0.23240438])

    plot_policies(P1, q1, EPSILON, 30, 30, 30, list(range(1, 5)), 'P1.png')
    plot_policies(P2, q2, EPSILON, 30, 30, 30, list(range(1, 16)), 'P2.png')
    plot_policies(P3, q3, EPSILON, 30, 30, 30, [1,2,3,4,7,8,9,10], 'P3.png')
    plot_policies(P4, q4, EPSILON, 150, 30, 30, [5, 10, 14, 15, 16, 17,
                  20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150], 'P4.png')
