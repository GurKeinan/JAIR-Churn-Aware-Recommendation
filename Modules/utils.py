import numpy as np
from collections.abc import Sequence


def drop_controlled_actions(prob_mat):
    """
    Drop actions that are controlled by other actions.
    """
    controlled_actions = []
    for action1 in range(len(prob_mat)):
        for action2 in range(action1+1, len(prob_mat)):
            if np.all(prob_mat[action1] >= prob_mat[action2]):
                controlled_actions.append(action2)
            elif np.all(prob_mat[action2] >= prob_mat[action1]):
                controlled_actions.append(action1)

    controlled_actions = list(set(controlled_actions))
    prob_mat_changed = np.delete(prob_mat, controlled_actions, axis=0)
    return prob_mat_changed


class SparseRepeatedArray(Sequence):
    def __init__(self, array):
        if not isinstance(array, np.ndarray):
            array = np.array(array, dtype=int)

        self.array = {}
        if len(array) > 0:
            self.array[0] = array[0]
            if len(array) > 1:
                mask = np.diff(array) != 0
                for i, b in enumerate(mask):
                    if b:
                        self.array[i+1] = array[i+1]

        self.last_idx = len(array) - 1
        self.num_values = len(self.array)

        self.keys = sorted(self.array.keys())
        self.values = [self[k] for k in self.keys]

        try:
            assert len(self.keys) == np.unique(self.keys).shape[0]
        except AssertionError:
            print("Non-unique actions encountered!")  # never happened

    @property
    def items(self):
        return zip(self.keys, self.values)

    def __len__(self):
        return self.last_idx + 1

    def __getitem__(self, item: int):
        if len(self) == 0:
            raise ValueError('Trying to getitem from an empty array.')
        if len(self) <= item:
            raise IndexError('Index outside of range.')
        if item < 0:
            raise IndexError('The index must be positive.')

        if item in self.array:
            return self.array[item]

        for k in self.keys:
            if k > item:
                break
            new_item = k
        return self.array[new_item]

    def __str__(self):
        lst = []
        if len(self) > 1:
            for k1, k2 in zip(self.keys[:-1], self.keys[1:]):
                if k2-1 > k1:
                    lst.append(f'Steps {k1}-{k2-1}: action {self[k1]}')
                else:
                    lst.append(f'Step {k1}: action: {self[k1]}')
        if len(self) > 0:
            lst.append(f'Steps {self.keys[-1]}+: action {self.values[-1]}')
        return '; '.join(lst)

    def normalize_and_interpolate(self, n_points=101):
        pass


if __name__ == '__main__':
    arr = SparseRepeatedArray([])
    try:
        print(arr[0])
    except ValueError as error:
        print(error)

    arr = [0, 0, 0, 5, 1, 1, 1, 1]
    arr = SparseRepeatedArray(arr)
    print(len(arr))
    print(arr)

    try:
        print(arr[-1])
    except IndexError as error:
        print(error)
    try:
        print(arr[100])
    except IndexError as error:
        print(error)

    lst = []
    for x in arr:
        lst.append(x)
    print(lst)

    lst = []
    for x in reversed(arr):
        lst.append(x)
    print(lst)

    print(arr.count(0))
    print(arr.__contains__(0))
