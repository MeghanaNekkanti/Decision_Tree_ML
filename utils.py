import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):

    total_sum = np.sum(branches)
    entropy_val = 0
    sum_arr = [sum(x) for x in branches]
    for i in range(len(branches)):
        entropy_val += (sum_arr[i] / total_sum) * entropy(branches[i])
    return S - entropy_val


def entropy(one_class):

    sum_att = 0
    entropy_one = 0
    for i in one_class:
        sum_att += i
    if sum_att == 0:
        entropy_one = 0
    else:
        for i in one_class:
            temp = i / sum_att
            if temp != 0:
                entropy_one += (-temp * np.log2(temp))
    return entropy_one

# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    raise NotImplementedError


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t',
                       deep=deep + 1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
