import numpy as np

import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    def compare_gain(self, gain, max_g, val_att, idx_attr):
        split = self.dim_split
        feature_uniq = self.feature_uniq_split
        if gain > max_g:
            max_g = gain
            split = idx_attr
            feature_uniq = val_att
        elif gain == max_g:
            if len(val_att) > len(self.feature_uniq_split):
                split = idx_attr
                feature_uniq = val_att
            elif len(val_att) == len(self.feature_uniq_split):
                if idx_attr < self.dim_split:
                    split = idx_attr
                    feature_uniq = val_att
        return max_g, split, feature_uniq

    # TODO: try to split current node
    def split(self):

        count = []
        temp = []
        max_i_g = -1
        d = {}

        if not all(v for v in self.features):
            self.splittable = False

        rows_count, features_count = np.shape(self.features)
        list_trans = list(self.features)
        list_trans = np.transpose(list_trans).tolist()
        for att_index in range(features_count):
            att_value = list(np.unique(list_trans[att_index]))
            att_value = sorted(att_value)
            for x in range(len(att_value)):
                d[att_value[x]] = {}
                for y in np.unique(self.labels):
                    count_label = 0
                    for j in range(rows_count):
                        if att_value[x] == list_trans[att_index][j] and y == self.labels[j]:
                            count_label += 1
                    d[att_value[x]][y] = count_label
            for x in att_value:
                for y in np.unique(self.labels):
                    temp.append(d.get(x).get(y))
                count.append(temp)
                temp = []
            labels = list(self.labels)
            labels = np.array(labels)
            unique_values, count_labels = np.unique(labels, return_counts=True)
            count_labels = count_labels.tolist()
            s = Util.entropy(count_labels)
            i_g = Util.Information_Gain(s, count)
            # print("entropy, IG", s, i_g, count)
            max_i_g, self.dim_split, self.feature_uniq_split = self.compare_gain(i_g, max_i_g, att_value, att_index)
            count = []

        if max_i_g == 0:
            self.splittable = False

        feature_arr = np.array(self.features)
        for i in self.feature_uniq_split:
            new = np.where(feature_arr[:, self.dim_split] == i)
            j = new[0].tolist()
            new_features = feature_arr[j].tolist()
            new_labels = np.array(self.labels)[j].tolist()
            for x in new_features:
                del x[self.dim_split]
            # print("Splitting these now", new_features, new_labels)
            num_cls = np.unique(new_labels).size
            temp_node = TreeNode(new_features, new_labels, num_cls)
            if np.array(new_features).size == 0 or not all(v for v in new_features):
                temp_node.splittable = False
            self.children.append(temp_node)
        #  print(self.children, "children")

        for temp in self.children:
            if temp.splittable:
                temp.split()
        return

        # TODO: predict the branch or the class

    def predict(self, feature):

        child = self.feature_uniq_split.index(feature[self.dim_split])
        # print(child, self.children[child].splittable)
        if self.children[child].splittable:
            del feature[self.dim_split]
            return self.children[child].predict(feature)
        else:
            #  print(self.children[child].cls_max)
            return self.children[child].cls_max
