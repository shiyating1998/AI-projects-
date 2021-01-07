import csv
import math
from anytree import Node
from anytree.exporter import DotExporter
import matplotlib.pyplot as plt

FEATURES = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
            "free_sulfur_dioxide", "total_sulfur_dioxide",
            "density", "pH", "sulphates", "alcohol"]
training_examples = []


class WineExample:
    def __init__(self):
        self.fixed_acidity = -1
        self.volatile_acidity = -1
        self.citric_acid = -1
        self.residual_sugar = -1
        self.chlorides = -1
        self.free_sulfur_dioxide = -1
        self.total_sulfur_dioxide = -1
        self.density = -1
        self.pH = -1
        self.sulphates = -1
        self.alcohol = -1

        self.quality = -1

    def __str__(self):
        str = f'\t{self.fixed_acidity},{self.volatile_acidity},{self.citric_acid},{self.residual_sugar},{self.chlorides},{self.free_sulfur_dioxide},{self.total_sulfur_dioxide},{self.density},{self.pH},{self.sulphates},{self.alcohol},quality:{self.quality}.'
        return str

    def get_feature_value(self, feature):
        if feature == "fixed_acidity":
            return self.fixed_acidity
        elif feature == "volatile_acidity":
            return self.volatile_acidity
        elif feature == "citric_acid":
            return self.citric_acid
        elif feature == "residual_sugar":
            return self.residual_sugar
        elif feature == "chlorides":
            return self.chlorides
        elif feature == "free_sulfur_dioxide":
            return self.free_sulfur_dioxide
        elif feature == "total_sulfur_dioxide":
            return self.total_sulfur_dioxide
        elif feature == "density":
            return self.density
        elif feature == "pH":
            return self.pH
        elif feature == "sulphates":
            return self.sulphates
        elif feature == "alcohol":
            return self.alcohol
        elif feature == "quality":
            return self.quality

        return -1


# read_data(file_path): reads in the data
# input: file_path, the file path of the data we want to read
# output:
def read_data(file_path):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                line_count += 1
                example = WineExample()
                example.fixed_acidity = float(row[0])
                example.volatile_acidity = float(row[1])
                example.citric_acid = float(row[2])
                example.residual_sugar = float(row[3])
                example.chlorides = float(row[4])
                example.free_sulfur_dioxide = float(row[5])
                example.total_sulfur_dioxide = float(row[6])
                example.density = float(row[7])
                example.pH = float(row[8])
                example.sulphates = float(row[9])
                example.alcohol = float(row[10])

                example.quality = int(row[11])
                training_examples.append(example)
                # print(example)

        # print(f'Processed {line_count} lines.')
    return 0


def sort_examples(examples, feature):
    sorted_examples = []
    if feature == "fixed_acidity":
        sorted_examples = sorted(examples, key=lambda example: example.fixed_acidity)
    elif feature == "volatile_acidity":
        sorted_examples = sorted(examples, key=lambda example: example.volatile_acidity)
    elif feature == "citric_acid":
        sorted_examples = sorted(examples, key=lambda example: example.citric_acid)
    elif feature == "residual_sugar":
        sorted_examples = sorted(examples, key=lambda example: example.residual_sugar)
    elif feature == "chlorides":
        sorted_examples = sorted(examples, key=lambda example: example.chlorides)
    elif feature == "free_sulfur_dioxide":
        sorted_examples = sorted(examples, key=lambda example: example.free_sulfur_dioxide)
    elif feature == "total_sulfur_dioxide":
        sorted_examples = sorted(examples, key=lambda example: example.total_sulfur_dioxide)
    elif feature == "density":
        sorted_examples = sorted(examples, key=lambda example: example.density)
    elif feature == "pH":
        sorted_examples = sorted(examples, key=lambda example: example.pH)
    elif feature == "sulphates":
        sorted_examples = sorted(examples, key=lambda example: example.sulphates)
    elif feature == "alcohol":
        sorted_examples = sorted(examples, key=lambda example: example.alcohol)
    elif feature == "quality":
        sorted_examples = sorted(examples, key=lambda example: example.quality)
    return sorted_examples


# get_splits(examples, feature): given a set of examples and a feature, get all the
# possible split point values for the feature
# Input: examples - list of strings, feature - string
# Output:
def get_splits(examples, feature):
    sorted_examples = sort_examples(examples, feature)
    split_points = []

    count = len(sorted_examples)
    i = 0
    Lx = []
    Ly = []

    ex = sorted_examples[i]
    current_feature_value = ex.get_feature_value(feature)
    Lx.append(ex.quality)

    while current_feature_value == ex.get_feature_value(feature):
        if ex.quality != Lx[0]:
            Lx.append(ex.quality)
        i += 1
        if i == count:
            break
        ex = sorted_examples[i]

    prev_feature_value = current_feature_value
    current_feature_value = ex.get_feature_value(feature)
    Ly.append(ex.quality)

    if Lx != Ly and i == count:
        split_points.append((prev_feature_value + current_feature_value) / 2)
    #  print("low:", prev_feature_value, "high:", current_feature_value)
    while i < count:
        while current_feature_value == ex.get_feature_value(feature):
            if ex.quality != Ly[0]:
                Ly.append(ex.quality)

            i += 1
            if i == count:
                break
            ex = sorted_examples[i]

        if Lx != Ly:
            split_points.append((prev_feature_value + current_feature_value) / 2)
        #  print("low:", prev_feature_value, "high:", current_feature_value)
        #  print(split_points)
        prev_feature_value = current_feature_value
        current_feature_value = ex.get_feature_value(feature)
        Lx = Ly.copy()
        Ly = [ex.quality]
    # print(len(split_points))
    return split_points


# calc_entropy(examples): calculate the entropy of the probability distribution represented by the examples
# Input: examples - list of strings
# Output: entropy - integer
def calc_entropy(examples):
    num = 0
    entropy = 0
    if len(examples) > 0:
        c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        num = 0
        for i in examples:
            c[i.quality] += 1
            num += 1
        entropy = 0
        # print("num is:", num)

        for i in range(11):
            # print(i)
            # print("c[i]:", c[i])
            x = c[i] / num
            if num == 0:
                print("??")
            if x != 0:
                entropy += x * (math.log(x, 2))
            #   print("entropy cur is:", entropy)
        entropy *= -1

    #  print("entropy is:", entropy)

    return num, entropy


# calc_entropy_set(examples,total): calculate the entropy of the given subset, given a total number of examples
# Input: examples, total (int)
# Output: entropy (float)
def calc_entropy_set(examples, total):
    t, entropy = calc_entropy(examples)
    entropy *= t / total
    return entropy


# split_examples(examples, feature, split): given a split value for the given feature,
# split the examples into two sets, above the split value, and below the split value
# Input: examples - list of strings, feature - string, split - integer
# Output: set1 - list of examples above the split value, set2 - list of examples below the split value
def split_examples(examples, feature, split):
    set1 = []
    set2 = []
    # sort the examples based on feature
    # perform the split
    # if examples.feature < split, append to set1, else append to set2
    sorted_examples = sort_examples(examples, feature)
    for n in sorted_examples:
        if n.get_feature_value(feature) < split:
            set1.append(n)
        else:
            set2.append(n)
    return set1, set2


# choose_split(examples, feature): given a set of examples and a feature, return the split value
# that results in the largest expected information gain
# Input: examples - list of strings, feature - string
# Output: max_info_gain - float, split value - float
def choose_split(examples, feature):
    # first find all possible split points
    max_info_gain = -1
    split_val = -1
    split_points = get_splits(examples, feature)
    for split in split_points:
        total, entropy_before = calc_entropy(examples)
        set1, set2 = split_examples(examples, feature, split)
        entropy_after = calc_entropy_set(set1, total) + calc_entropy_set(set2, total)
        info_gain = entropy_before - entropy_after
        if info_gain > max_info_gain:
            split_val = split
            max_info_gain = info_gain

    # calculate the info gain of each split point
    # return the split point with largest info gain

    return max_info_gain, split_val


# choose_feature(example, features): given a set of examples and a set of features,
# choose a feature and a split value for the feature that maximizes the expected information gain
# Input: examples - list of strings, features - string
# Output: feature, split value
def choose_feature(examples, features):
    # for each feature, find the feature with the greatest info gain
    # for each feature, find the greatest split point for infogain
    max_info_gain = -1
    split_val = -1
    f = ""
    for feature in features:
        info_gain, split = choose_split(examples, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            split_val = split
            f = feature
    return split_val, f


# are_examples_same_class(examples): given a set of examples, return True if they are all in the
# same class, False otherwise
# Input: examples (set of examples)
# Ouput: Boolean
def are_examples_same_class(examples):
    flag = True
    q = examples[0].quality
    for n in examples:
        if q != n.quality:
            flag = False
            break
    return flag


def majority(examples):
    c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in examples:
        c[i.quality] += 1
    count = -1
    q = -1
    for i in range(11):
        # print(i)
        # print("c[i]:", c[i])
        if c[i] > count:
            count = c[i]
            q = i
    return q


def edgeattrfunc(node, child):
    return 'label="%s"' % (child.label)


# depth(tree): find the depth of the tree
# input: tree
# output: depth (int)
def depth(tree):
    if len(tree.children) == 0:
        return 1

    left = depth(tree.children[0])
    right = depth(tree.children[1])

    return max(left, right) + 1


# learn_dt(tree, examples, features): given a set of examples and a set of features
# learn a decision tree to classify the examples
# Input: tree, examples, features
# Output: tree
def learn_dt(tree, examples, features, max_depth, depth):
    if are_examples_same_class(examples):
        return Node(examples[0].quality, parent=tree, data=examples, )
    elif depth == max_depth:
        return Node(majority(examples), parent=tree, data=examples)
    else:
        split, f = choose_feature(examples, features)
        set1, set2 = split_examples(examples, f, split)
        # recurse on the subtrees
        subtree1 = learn_dt(tree, set1, features, max_depth, depth + 1)
        subtree1.label = f + " val <=" + str(split)
        subtree2 = learn_dt(tree, set2, features, max_depth, depth + 1)
        subtree2.label = f + " val >" + str(split)

        tree = Node(name=f + " " + str(split), children=[subtree1, subtree2, ], feature=f,
                    split=split, data=examples)

    return tree


# learn_dt_infogain(tree, examples, features, can_be_prune): given a set of examples and a set of features
# learn a decision tree to classify the examples
# Input: tree, examples, features
# Output: tree
def learn_dt_min_infogain(tree, examples, features, can_be_prune):
    if len(examples) == 0:
        return can_be_prune, majority(tree.data)
    elif are_examples_same_class(examples):
        return can_be_prune, Node(examples[0].quality, parent=tree, data=examples)
    else:
        split, f = choose_feature(examples, features)
        set1, set2 = split_examples(examples, f, split)
        # recurse on the subtrees
        subtree1 = learn_dt_min_infogain(tree, set1, features, can_be_prune)[1]
        subtree1.label = f + " val <=" + str(split)
        subtree2 = learn_dt_min_infogain(tree, set2, features, can_be_prune)[1]
        subtree2.label = f + " val >" + str(split)

        tree = Node(name=f + " " + str(split), children=[subtree1, subtree2, ], data=examples,
                    feature=f, split=split)  # TODO how to reduce features?
        if depth(subtree1) <= 1 and depth(subtree2) <= 1:
            can_be_prune.append(tree)
    # print(can_be_prune)
    # print("number of nodes:", len(can_be_prune))
    # print(can_be_prune[0].data)

    return can_be_prune, tree


# prune_min_info_gain: prune the tree with the given min_info_gain
# input: tree, can_be_prune(set of nodes that can possibly be prune), min_info_gain - threshold for pruning
# output: pruned tree
def prune_min_info_gain(tree, can_be_prune, min_info_gain):
    cc = []
    for n in can_be_prune:
        if n.parent:
            #info_gain, split = choose_split(n.data, n.feature)
            if choose_split(n.data, n.feature)[0] < min_info_gain:
                n.children = []
                n.name = majority(n.data)
                p = n.parent
                if p and depth(p) <= 2:
                    cc.append(p)
                    can_be_prune.append(p)
    return cc, tree


# predict(tree, example, max_depth): given a decision tree and an example, return the class label
# for the example
# Input: tree, example
# Output: class label
def predict(tree, example, max_depth):
    if str(tree.name).isnumeric():
        return tree.name
    else:
        if tree.depth + 1 == max_depth:
            return majority(tree.data)
        elif example.get_feature_value(tree.feature) <= float(tree.split):
            return predict(tree.children[0], example, max_depth)
        else:
            return predict(tree.children[1], example, max_depth)


# get_prediction_accuracy(tree,data): given a decision tree and a data-set, return the
# prediction accuracy of the decision tree on the data-set
# Input: tree, data
# Output: prediction accuracy
def get_prediction_accuracy(tree, data, max_d):
    total = 0.0
    correct = 0.0
    for n in data:
        total += 1
        predict_value = predict(tree, n, max_d)
        if predict_value == n.quality:
            correct += 1
    if total != 0:
        return correct / total
    else:
        return 1



# cross_validation(data,file_path): perform five-fold cross-validation on the provided data
# set, saves a plot of the training accuracy and validation accuracy in the file at the file_path
# returns the best maximum depth of the decision tree
# Output: best_maximum_depth integer
def cross_validation(data, file_path):
    fold1 = data[:319]
    fold2 = data[319:638]
    fold3 = data[638:957]
    fold4 = data[957:1276]
    fold5 = data[1276:]

    max_d = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    accuracy_data_validation = []
    accuracy_data_training = []
    d = 4

    best_depth = -1
    best_accuracy = -1

    training_set = fold1 + fold2 + fold3 + fold4
    t1 = learn_dt(Node("root"), training_set, FEATURES, 100, 1)

    training_set = fold1 + fold2 + fold3 + fold5
    t2 = learn_dt(Node("root"), training_set, FEATURES, 100, 1)

    training_set = fold1 + fold2 + fold4 + fold5
    t3 = learn_dt(Node("root"), training_set, FEATURES, 100, 1)

    training_set = fold1 + fold3 + fold4 + fold5
    t4 = learn_dt(Node("root"), training_set, FEATURES, 100, 1)

    training_set = fold2 + fold3 + fold4 + fold5
    t5 = learn_dt(Node("root"), training_set, FEATURES, 100, 1)

    while d < 13:
        accuracy_validation = 0
        accuracy_training = 0

        training_set = fold1 + fold2 + fold3 + fold4
        validation_set = fold5
        accuracy_validation += get_prediction_accuracy(t1, validation_set, d)
        accuracy_training += get_prediction_accuracy(t1, training_set, d)

        training_set = fold1 + fold2 + fold3 + fold5
        validation_set = fold4
        accuracy_validation += get_prediction_accuracy(t2, validation_set, d)
        accuracy_training += get_prediction_accuracy(t2, training_set, d)

        training_set = fold1 + fold2 + fold4 + fold5
        validation_set = fold3
        accuracy_validation += get_prediction_accuracy(t3, validation_set, d)
        accuracy_training += get_prediction_accuracy(t3, training_set, d)

        training_set = fold1 + fold3 + fold4 + fold5
        validation_set = fold2
        accuracy_validation += get_prediction_accuracy(t4, validation_set, d)
        accuracy_training += get_prediction_accuracy(t4, training_set, d)

        training_set = fold2 + fold3 + fold4 + fold5
        validation_set = fold1
        accuracy_validation += get_prediction_accuracy(t5, validation_set, d)
        accuracy_training += get_prediction_accuracy(t5, training_set, d)

        average_accuracy_validation = accuracy_validation / 5
        average_accuracy_training = accuracy_training / 5
        # print("depth:", d)
        # print("average validation:", average_accuracy_validation)
        # print("Average training:", average_accuracy_training)
        accuracy_data_validation.append(average_accuracy_validation)
        accuracy_data_training.append(average_accuracy_training)

        if best_accuracy < average_accuracy_validation:
            best_accuracy = average_accuracy_validation
            best_depth = d
        d += 1

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(max_d, accuracy_data_training, label="average prediction accuracy on training set")
    ax.plot(max_d, accuracy_data_validation, label="average prediction accuracy on validation set")
    ax.legend()
    ax.set_xlabel("Maximum Depth")
    ax.set_ylabel("Average Prediction Accuracy")
    ax.set_title("Prediction Accuracy vs Maximum Depth")
    for i_x, i_y in zip(max_d, accuracy_data_training):
        plt.text(i_x, i_y, '({}, {})'.format(i_x, '{0:.3f}'.format(i_y)))
    for i_x, i_y in zip(max_d, accuracy_data_validation):
        plt.text(i_x, i_y, '({}, {})'.format(i_x, '{0:.3f}'.format(i_y)))
    plt.savefig(file_path)
    return best_depth


# cross_validation2(data,file_path): perform five-fold cross-validation on the provided data
# set, saves a plot of the training accuracy and validation accuracy in the file at the file_path
# returns the min_info_gain of the decision tree that has the best performance on validation_set
# Output: min_info_gain
def cross_validation2(data, file_path):
    fold1 = data[:319]
    fold2 = data[319:638]
    fold3 = data[638:957]
    fold4 = data[957:1276]
    fold5 = data[1276:]

    min_i = []
    a = 0.2
    while a <= 1:
        min_i.append(a)
        a += 0.1

    accuracy_data_validation = []
    accuracy_data_training = []
    i = 0.2

    best_min = 2
    best_accuracy = -1

    training_set = fold1 + fold2 + fold3 + fold4
    can_be_prune1, t1 = learn_dt_min_infogain(Node("root"), training_set, FEATURES, [])

    training_set = fold1 + fold2 + fold3 + fold4
    can_be_prune2, t2 = learn_dt_min_infogain(Node("root"), training_set, FEATURES, [])

    training_set = fold1 + fold2 + fold3 + fold4
    can_be_prune3, t3 = learn_dt_min_infogain(Node("root"), training_set, FEATURES, [])

    training_set = fold1 + fold2 + fold3 + fold4
    can_be_prune4, t4 = learn_dt_min_infogain(Node("root"), training_set, FEATURES, [])

    training_set = fold1 + fold2 + fold3 + fold4
    can_be_prune5, t5 = learn_dt_min_infogain(Node("root"), training_set, FEATURES, [])

    while i <= 1:
        # print("min-info-gain:", i)
        accuracy_validation = 0
        accuracy_training = 0

        training_set = fold1 + fold2 + fold3 + fold4
        validation_set = fold5
        can_be_prune1, t1 = prune_min_info_gain(t1, can_be_prune1, i)
        accuracy_validation += get_prediction_accuracy(t1, validation_set, 100)
        accuracy_training += get_prediction_accuracy(t1, training_set, 100)

        training_set = fold1 + fold2 + fold3 + fold5
        validation_set = fold4
        can_be_prune2, t2 = prune_min_info_gain(t2, can_be_prune2, i)
        accuracy_validation += get_prediction_accuracy(t2, validation_set, 100)
        accuracy_training += get_prediction_accuracy(t2, training_set, 100)

        training_set = fold1 + fold2 + fold4 + fold5
        validation_set = fold3
        can_be_prune3, t3 = prune_min_info_gain(t3, can_be_prune3, i)
        accuracy_validation += get_prediction_accuracy(t3, validation_set, 100)
        accuracy_training += get_prediction_accuracy(t3, training_set, 100)

        training_set = fold1 + fold3 + fold4 + fold5
        validation_set = fold2
        can_be_prune4, t4 = prune_min_info_gain(t4, can_be_prune4, i)
        accuracy_validation += get_prediction_accuracy(t4, validation_set, 100)
        accuracy_training += get_prediction_accuracy(t4, training_set, 100)

        training_set = fold2 + fold3 + fold4 + fold5
        validation_set = fold1
        can_be_prune5, t5 = prune_min_info_gain(t5, can_be_prune5, i)
        accuracy_validation += get_prediction_accuracy(t5, validation_set, 100)
        accuracy_training += get_prediction_accuracy(t5, training_set, 100)

        average_accuracy_validation = accuracy_validation / 5
        average_accuracy_training = accuracy_training / 5
        # print("average validation:", average_accuracy_validation)
        # print("Average training:", average_accuracy_training)
        accuracy_data_validation.append(average_accuracy_validation)
        accuracy_data_training.append(average_accuracy_training)

        if best_accuracy < average_accuracy_validation:
            best_accuracy = average_accuracy_validation
            best_min = i
        i += 0.1

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(min_i, accuracy_data_training, label="average prediction accuracy on training set")
    ax.plot(min_i, accuracy_data_validation, label="average prediction accuracy on validation set")
    ax.legend()
    ax.set_xlabel("Min Information Gain")
    ax.set_ylabel("Average Prediction Accuracy")
    ax.set_title("Prediction Accuracy vs Min Information Gain")
    plt.savefig(file_path)
    return best_min


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = "winequality_red_comma.csv"
    tree_full = read_data(file_path)

    # part A  (1) generate tree-full.png
    t = learn_dt(Node("root", level=1), training_examples, FEATURES, 100, 1)
    DotExporter(t, edgeattrfunc=edgeattrfunc).to_picture("tree-full.png")
    # (2) depth of the tree-full
    print("depth of tree-full:", depth(t))
    # (3) prediction accuracy of tree-tree
    prediction_accuracy = get_prediction_accuracy(t, training_examples, 100)
    print("prediction accuracy of tree-full:", prediction_accuracy)

    # part B (1) generate plot
    max_d = cross_validation(training_examples, "cv-max-depth.png")
    # (2) the best value of the maximum depth of the tree is 6
    print("max_depth:", max_d)
    # (3) generate tree-max-depth.png with max-depth=6
    t = learn_dt(Node("root"), training_examples, FEATURES, max_d, 1)
    DotExporter(t, edgeattrfunc=edgeattrfunc).to_picture("tree-max-depth.png")
    # (4) prediction accuracy of tree-max-depth on the entire data-set
    prediction_accuracy = get_prediction_accuracy(t, training_examples, max_d)
    print("prediction accuracy of tree-max-depth:", prediction_accuracy)

    # part C (1) generate the plot cv-min-info-gain.png using cross validation
    min_info_gain = cross_validation2(training_examples, "cv-min-info-gain.png")

    # (2) the best value of the min_info_gain is:
    print("min_info_gain is at:", min_info_gain)

    # (3) the decision tree generated using the entire data-set of the tree-min-info-gain
    can_be_prune, t = learn_dt_min_infogain(Node("root"), training_examples, FEATURES, [])
    t = prune_min_info_gain(t, can_be_prune, min_info_gain)
    DotExporter(t, edgeattrfunc=edgeattrfunc).to_picture("tree-min-info-gain.png")

    # (4) the prediction accuracy of the decision tree on the entire dataset is:
    prediction_accuracy = get_prediction_accuracy(t, training_examples, 100)
    print("prediction accuracy of tree-min-info-gain:", prediction_accuracy)
