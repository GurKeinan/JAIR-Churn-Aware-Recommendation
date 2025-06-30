import subprocess
import os
import numpy as np
import pathlib

root_path = pathlib.Path(__file__).parent.parent.absolute()
POMDPSOL_PATH = os.path.join(root_path, 'sarsop', 'src', 'pomdpsol')
# cygwin_bin_path = "path-to-cygwin-bin"  # Modify this to the path of the cygwin bin folder
# CYGWIN_ENV = os.environ.copy()
# CYGWIN_ENV["PATH"] = cygwin_bin_path + ";" + CYGWIN_ENV["PATH"]
CYGWIN_ENV = os.environ.copy()  # Just keep the environment as is on macOS

class TransitionModel:
    def __init__(self, P, q, abs_idx, disc_idx, n_states):
        self.abs_idx, self.disc_idx, self.n_states = abs_idx, disc_idx, n_states
        self.n_types = len(q)
        self.q = q
        self.P = P
        self.gamma = P.max()

    def probability(self, next_state, state, action):
        dist = self.get_distribution(state, action)
        return dist[next_state]

    def get_all_states(self):
        return list(range(self.n_states))

    def get_all_actions(self):
        return list(range(self.P.shape[0]))

    def get_distribution(self, state, action):
        dist = [0.] * self.n_states
        if state == self.abs_idx:
            # absorbing state
            dist[self.abs_idx] = 1.
        elif state >= self.disc_idx:
            # arbitrary non-initial state
            p_remain = self.P[action, state - self.disc_idx] / self.gamma
            dist[state] = p_remain
            dist[self.abs_idx] = 1. - p_remain
        else:
            # initial state (sampled from q)
            p_remain = self.P[action, state]
            dist[state + self.disc_idx] = p_remain
            dist[self.abs_idx] = 1. - p_remain
        return dist


def to_pomdp_file(init_belief, transition_model, abs_idx, disc_idx, output_path=None, discount_factor=0.95, float_precision=9):
    """
    Pass in an Agent, and use its components to generate
    a .pomdp file to `output_path`.

    The .pomdp file format is specified at:
    http://www.pomdp.org/code/pomdp-file-spec.html
    """
    n_states = len(transition_model.get_all_states())
    n_actions = len(transition_model.get_all_actions())

    content = f"discount: %.{float_precision}f\n" % discount_factor
    content += "values: reward\n"  # We only consider reward, not cost.

    content += "states: %s\n" % n_states
    content += "actions: %s\n" % n_actions

    list_of_observations = "True False"
    content += "observations: %s\n" % list_of_observations

    # Starting belief state - they need to be normalized
    content += "start: %s\n" % (
        " ".join(
            [
                f"%.{float_precision}f" % init_belief[s]
                for s in range(n_states)
            ]
        )
    )

    # State transition probabilities - they need to be normalized
    content += "T : * : * : * 0\n"
    for s in range(n_states):
        if s == abs_idx:
            continue
        for a in range(n_actions):
            s_next = s if s >= disc_idx else s + disc_idx
            prob = transition_model.probability(s_next, s, a)
            content += f"T : %s : %s : %s %.{float_precision}f\n" % (
                a,
                s,
                s_next,
                prob,
            )
            content += f"T : %s : %s : %s %.{float_precision}f\n" % (
                a,
                s,
                abs_idx,
                1-prob,
            )
    content += f"T : * : {abs_idx} : {abs_idx} 1\n"

    # Observation probabilities - they need to be normalized
    content += "O : * : * : True 1\n"
    content += "O : * : * : False 0\n"
    content += f"O : * : {abs_idx} : True 0\n"
    content += f"O : * : {abs_idx} : False 1\n"

    # Immediate rewards
    content += "R : * : * : * : * 1\n"
    content += f"R : * : * : {abs_idx} : * 0\n"

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(content)


def sarsop(
    P, q,
    timeout=60,
    memory=100,
    precision=1e-6,
    pomdp_name="RECAPC",
    remove_generated_files=True,
    verbose=False,
):
    """
    SARSOP, using the binary from https://github.com/AdaCompNUS/sarsop
    This is an anytime POMDP planning algorithm
    """
    n_types = P.shape[1]

    abs_idx = 2 * n_types
    disc_idx = n_types
    n_states = 2 * n_types + 1

    init_belief = {i: 0. for i in range(n_states)}
    for i in range(n_types):
        init_belief[i] = q[i]

    gamma = P.max()
    transition_model = TransitionModel(P, q, abs_idx, disc_idx, n_states)

    stdout = subprocess.PIPE
    stderr = subprocess.STDOUT

    modules_path = pathlib.Path(__file__).parent.absolute()

    path = os.path.join(modules_path, f"{pomdp_name}.pomdp")
    pomdp_path = path + '.pomdp'
    policy_path = path + '.policy'
    to_pomdp_file(init_belief, transition_model, abs_idx, disc_idx, pomdp_path, discount_factor=gamma)

    proc = subprocess.run(
        [
            POMDPSOL_PATH,
            "--timeout",
            str(timeout),
            "--memory",
            str(memory),
            "--precision",
            str(precision),
            "--output",
            policy_path,
            pomdp_path,
        ],
        env=CYGWIN_ENV,
        stdout=stdout,
        stderr=stderr
    )
    stdout_text = proc.stdout.decode()
    if verbose:
        print(f"b'{stdout_text}'")

    # Remove temporary files
    if remove_generated_files:
        os.remove(pomdp_path)
        os.remove(policy_path)

    log = f"'{stdout_text}'".split('\n')[-7].split()
    elapsed_time, n_backups, l_bound = eval(log[0]), eval(log[2]), eval(log[3])

    # sarsop only updates elapsed_time about every 15 ms, so adding randomness to simulate real time
    elapsed_time += np.random.uniform(high=7.5)/1000
    return elapsed_time, n_backups, l_bound


if __name__ == '__main__':
    from simulations import get_random_RECAPC, run_RECAPC
    np.random.seed(0)
    n_actions, n_types = 10, 10

    # P, q = get_random_RECAPC(n_actions, n_types, q_std=.5)
    # run_RECAPC(P, q)
    # elapsed_time, n_backups, l_bound = sarsop(P, q, verbose=True)
    # print(f'sarsop number of backups: {n_backups}, lower bound: {l_bound}')

    n_trials = 25
    explored_branches, n_backups = 0, 0
    for i in range(n_trials):
        P, q = get_random_RECAPC(n_actions, n_types, q_std=.5, verbose=False)
        val, _, _, branches, _, _ = run_RECAPC(P, q, verbose=False)
        elapsed_time, backups, l_bound = sarsop(P, q, verbose=False)
        print(f"\n_______________________{i}__________________________")
        print(f"branches B&B: {branches}, backups sarsop: {backups}")
        print(f"val B&B: {round(val, 5)}, sarsop: {l_bound}")
        explored_branches += branches
        n_backups += backups
    print(f"avg branches: {explored_branches / n_trials}, avg backups: {n_backups / n_trials}")
