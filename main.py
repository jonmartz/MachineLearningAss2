# Import scikit-learn dataset library
import pandas as pd
import numpy as np
import csv
import pickle
from sklearn import datasets
# Import train_test_split function
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from itertools import zip_longest
from joblib import dump, load
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer


def train_test_split(dataset, train_fraction):
    """
    Splits the dataset into a train and a test set
    :param dataset: data to be split
    :param categorical_cols: list of the column names of the categorical columns (previously identified automatically)
    :param train_fraction: portion of dataset to be used as train set
    :return: a list [train set, one-hot-encoded train set, test set, one-hot-encoded test set]
    """

    #### Default - set string values as categorial except target column   ####
    ##########################################################################
    categorical_cols = []
    for (columnName, columnData) in dataset.iteritems():
        for value in columnData.values:
            if type(value) is str:
                categorical_cols.append(columnName)
                break

    if (categorical_cols[-1] == "CLASS"):
        categorical_cols = np.delete(categorical_cols, -1)

    ## AR: improve categorial selction

    dataset_encoded = OneHotEncoder(cols=categorical_cols, use_cat_names=True).fit_transform(dataset)
    if (train_fraction == 1):
        return dataset_encoded, dataset_encoded, dataset_encoded, dataset_encoded

    train_len = int(len(dataset.index) * train_fraction)
    train_set = dataset.sample(n=train_len, random_state=1)
    train_set_encoded = dataset_encoded.loc[train_set.index].reset_index(drop=True)
    test_set = dataset.drop(train_set.index).reset_index(drop=True)
    test_set_encoded = dataset_encoded.drop(train_set.index).reset_index(drop=True)

    return train_set.reset_index(drop=True), train_set_encoded, test_set, test_set_encoded
    # return train_set.reset_index(drop=True), dataset_encoded, test_set, dataset_encoded


def rmissingvaluecol_onedf(dff, threshold):
    my_df = dff.copy()
    l = []
    l = list(my_df.drop(my_df.loc[:, list((100 * (my_df.isnull().sum() / len(my_df.index)) >= threshold))].columns,
                        1).columns.values)
    #    print("# Columns having more than %s percent missing values:"%threshold,(dff.shape[1] - len(l)))
    #    print("Columns:\n",list(set(list((dff.columns.values))) - set(l)))
    dff1 = my_df[l]
    return (dff1)
    # return l


def rmissingvaluecol_twodf(dff_train, dff_test):
    my_train_df = dff_train.copy()
    my_test_df = dff_test.copy()

    l_removed_columns = []

    for value in my_train_df.columns.values:
        if value not in my_test_df.columns.values:
            if (value != "CLASS"):
                # print ("missing value is:", value )
                # print ("my_test_df.columns.values:", my_test_df.columns.values)
                # print ("my_train_df.columns.values:", my_train_df.columns.values)
                l_removed_columns.append(value)
                my_train_df.drop(value, axis=1, inplace=True)
            # my_train_df.drop(value)

    for value in my_test_df.columns.values:
        if value not in my_train_df.columns.values:
            if (value != "CLASS"):
                # print ("missing value is:", value )
                # print ("my_test_df.columns.values:", my_test_df.columns.values)
                # print ("my_train_df.columns.values:", my_train_df.columns.values)
                l_removed_columns.append(value)
                my_test_df.drop(value, axis=1, inplace=True)

    return my_train_df, my_test_df


def missing_values_impul(dff, my_strategy):
    my_df = dff.copy()
    col = my_df.columns.values

    my_df = my_df.fillna('')

    imp = SimpleImputer(missing_values='', strategy=my_strategy)
    my_df = imp.fit_transform(my_df)
    my_df2 = pd.DataFrame(my_df, columns=col)
    return my_df2


def drop_corr_feat(dff):
    my_df = dff.copy()
    l_numeric_cols = []

    ### AR: write for categorial value

    ### for numeric values
    for (columnName, columnData) in my_df.iteritems():
        for value in columnData.values:
            if not (type(value) is str):
                l_numeric_cols.append(columnName)
                break
    numeric_df = my_df[l_numeric_cols]
    for value in numeric_df.columns:
        numeric_df[value] = numeric_df[value].astype(float)

    corr_matrix = numeric_df.corr(method='pearson').abs()
    print("corr matrix", corr_matrix)
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print("to drop:", to_drop)

    my_df.drop(my_df[to_drop], axis=1)
    #    for (columnName, columnData) in my_df.iteritems():
    #        for value in columnData.values:
    #            if type(value) is str:
    #        ## write code fro strings\categorials
    #                print (columnName)
    #                break
    #            else:
    #                corr_matrix = my_df.corr().abs()
    #                # Select upper triangle of correlation matrix
    #                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #                # Find index of feature columns with correlation greater than 0.95
    #                to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]
    #
    #    print ("to drop:", to_drop)

    return my_df


def match_columns(dff_train, dff_test):
    l1 = []
    l1 = list(dff_train.columns)

    l2 = []
    l2 = list(dff_test.columns)

    print("l1 in match_columns is:\n", l1)
    print("l2 in match_columns is:\n", l2)

    for value in l1:
        if not value in l2:
            dff_train.drop(value)

    for value in l2:
        if not value in l1:
            dff_test.drop(value)

    return dff_train, dff_test


def test(parsed_train_set_encoded, parsed_test_set_encoded, target_col, clf, mode):
    x_train = parsed_train_set_encoded.drop(columns=target_col)
    y_train = parsed_train_set_encoded[target_col]

    if (mode == "train"):
        x_test = parsed_test_set_encoded.drop(columns=target_col)
        y_test = parsed_test_set_encoded[target_col]
        # print ("x_train is:", x_train)
        # print ("y_train is:", y_train)
        clf.fit(x_train, y_train)
        # dump(clf, r"C:\Users\yrakotch\Desktop\tmp\MasterDegree\LemidaHishuvit\exercise2\train.joblib")

    if (mode == "test"):
        x_test = parsed_test_set_encoded
        # print ("test parsed data is: ", x_test)
        clf.fit(x_train, y_train)
        # clf = load(r"C:\Users\yrakotch\Desktop\tmp\MasterDegree\LemidaHishuvit\exercise2\train.joblib")

    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)
    ### used only for experiments
    if (mode == "train"):
        print("Accuracy of train (the test portion in train set):", metrics.accuracy_score(y_test, y_pred))
        print("AUC in train set:", roc_auc_score(y_test, y_pred_proba[:, 1]))

    ProbToYes = y_pred_proba[:, 1]
    # print("length of x_test is:\n",len(x_test))
    # print("length of ProbToYes is:\n",len(ProbToYes))
    print("ProbToYes is:\n", ProbToYes)

    with open(r"C:\Users\yrakotch\Desktop\tmp\MasterDegree\LemidaHishuvit\exercise2\out.csv", mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Id", "ProbToYes"])
        for i in range(len(ProbToYes)):
            writer.writerow([int(i + 1), ProbToYes[i]])


# Choose a classifier
# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=450)
target_col = 'CLASS'

parsed_train_set_encoded = []
parsed_test_set_encoded = []
dataset = []
train_set = []
train_set_encoded_a = []
test_set = []
test_set_encoded_b = []
to_drop = []

# =============================================================================
# ##### OFFLINE testing code ####### train & validation
# dataset = pd.read_csv(r"C:\Users\yrakotch\Desktop\tmp\MasterDegree\LemidaHishuvit\exercise2\train.csv")
# train_fraction = 0.7
# ## remove columns with minimal data
# dataset =  rmissingvaluecol_onedf(dataset,80) #Here threshold is 10% which means we are going to drop columns having more than 10% of missing values
# ## deal with NaN pre-encoding
# #print("dataset pre impulation is:", dataset )
# strategy = 'most_frequent'
# dataset = missing_values_impul(dataset, strategy)
# #print("dataset post impulation is:", dataset)
# ## Drop correlated features
# dataset = drop_corr_feat(dataset)
#
# ## encoding
# train_set, train_set_encoded_a, test_set, test_set_encoded_b = train_test_split(dataset, train_fraction)
# #parsed_train_set_encoded = rmissingvaluecol_onedf(train_set_encoded_a,0.01) #Here threshold is 0.01% which means we are going to drop columns having more than 0.01% of missing values
# #parsed_test_set_encoded = rmissingvaluecol_onedf(test_set_encoded_b,0.01) #Here threshold is 0.01% which means we are going to drop columns having more than 0.01% of missing values
#
# ### removing columns to match train testing to actual testing
# #parsed_train_set_encoded, parsed_test_set_encoded_pre = rmissingvaluecol_twodf(parsed_train_set_encoded, parsed_train_set_encoded_pre, 0.01)
# #parsed_test_set_encoded, parsed_test_set_encoded_pre = rmissingvaluecol_twodf(parsed_test_set_encoded, parsed_test_set_encoded_pre, 0.01)
#
#
# ## AR: write code to drop dependant columns, save them and set as parameter to online test
#
#
# #print("\n for offline training train_set_encoded_a:\n", train_set_encoded_a)
# #print("\n for offline training test_set_encoded_b:\n", test_set_encoded_b)
# test(train_set_encoded_a, test_set_encoded_b, target_col, clf, "train")
#
# =============================================================================


##### ONLINE testing code ####### train & validation
dataset = pd.read_csv(r"C:\Users\yrakotch\Desktop\tmp\MasterDegree\LemidaHishuvit\exercise2\train.csv")
### if train_fraction = 1, just encode, no split
train_fraction = 1
## remove columns with minimal data
dataset = rmissingvaluecol_onedf(dataset,
                                 80)  # Here threshold is 10% which means we are going to drop columns having more than 10% of missing values
## deal with NaN pre-encoding

strategy = 'most_frequent'
dataset = missing_values_impul(dataset, strategy)

to_drop = ['A178', 'A182', 'A183', 'A189', 'A420', 'A422']
dataset.drop(dataset[to_drop], axis=1)
## encoding
train_set, train_set_encoded_a, test_set, test_set_encoded_b = train_test_split(dataset, train_fraction)
actual_train_set = train_set_encoded_a
# parsed_train_set_encoded = rmissingvaluecol_onedf(train_set_encoded_a,0.01) #Here threshold is 0.01% which means we are going to drop columns having more than 0.01% of missing values
# parsed_test_set_encoded = rmissingvaluecol_onedf(test_set_encoded_b,0.01) #Here threshold is 0.01% which means we are going to drop columns having more than 0.01% of missing values


dataset = pd.read_csv(r"C:\Users\yrakotch\Desktop\tmp\MasterDegree\LemidaHishuvit\exercise2\test.csv")
### if train_fraction = 1, just encode, no split
train_fraction = 1
## remove columns with minimal data
dataset = rmissingvaluecol_onedf(dataset,
                                 80)  # Here threshold is 10% which means we are going to drop columns having more than 10% of missing values
## deal with NaN pre-encoding

strategy = 'most_frequent'
dataset = missing_values_impul(dataset, strategy)

# Drop columns based on training set correlation
to_drop = ['A178', 'A182', 'A183', 'A189', 'A420', 'A422']
dataset.drop(dataset[to_drop], axis=1)

## encoding
train_set, train_set_encoded_a, test_set, test_set_encoded_b = train_test_split(dataset, train_fraction)
actual_test_set = train_set_encoded_a

### removing columns to match train testing to actual testing
actual_train_set, actual_test_set = rmissingvaluecol_twodf(actual_train_set, actual_test_set)

## AR: write code to drop dependant columns, save them and set as parameter to online test


test(actual_train_set, actual_test_set, target_col, clf, "test")

#



