import numpy as np


def test_np_random_choice():
    item_num      = 5
    resource_list = [i for i in range(25)]
    for i in range(5):
        dict_users = set(np.random.choice(resource_list, item_num, replace=False))
        print(dict_users)


if __name__ == "__main__":
    test_np_random_choice()
