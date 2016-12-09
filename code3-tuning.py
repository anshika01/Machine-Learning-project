import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import warnings
import numpy as np
warnings.filterwarnings("ignore")


############# data processor ############################################################

def change_categorical_data_type(df):
    for col in category_lst:
        df[col] = df[col].astype('category')
    return df


def latent_variable(destination_file):
    destinations = pd.read_csv(destination_file)  # Read in latent destinaton variables
    pca = PCA(n_components=3)
    dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
    dest_small = pd.DataFrame(dest_small)
    dest_small.columns = ['pca1', 'pca2', 'pca3']
    dest_small["srch_destination_id"] = destinations["srch_destination_id"]
    dest_small["new_srch_destination_id"] = destinations["srch_destination_id"]  # todo
    print "done"
    return dest_small


def delta_to_int(timedelta):
    try:
        return timedelta.days
    except AttributeError:
        return np.nan


def date_to_weekday(date):
    return date.weekday()


def calc_fast_features(df):
    df['dow'] = pd.to_datetime(df['date_time']).apply(date_to_weekday)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['srch_ci'] = pd.to_datetime(df['srch_ci'], errors='coerce')
    df['srch_co'] = pd.to_datetime(df['srch_co'], errors='coerce')
    df['stay_days'] = (df['srch_co'] - df['srch_ci'])
    df['before_days'] = (df['srch_ci'] - df['date_time'])
    df['stay_days'] = df['stay_days'].apply(delta_to_int)
    df['before_days'] = df['before_days'].apply(delta_to_int)
    dest_small = latent_variable(DEST_FILE)
    df1 = df.join(dest_small, on="new_srch_destination_id", how='left', rsuffix="dest") # todo
    df2 = df1.drop("new_srch_destination_iddest", axis=1) # todo
    #df3 = df2[feature_col]
    df3 = df2[selected_features]
    return df3


def training_processor(train):
    train = train.dropna(axis=0, how='any')
    train = change_categorical_data_type(train)

    train_y = train['hotel_cluster']
    train_x = calc_fast_features(train)

    imp = Imputer(strategy='mean', axis=0)
    train_x = imp.fit_transform(train_x)

    return train_x, train_y


def testing_processor(test):
    imp = Imputer(strategy='mean', axis=0)
    test_X = calc_fast_features(test)
    test_X = imp.fit_transform(test_X)
    test_Y = test['hotel_cluster']

    return test_X, test_Y


######### model selection ##########################################################################


def run_model_methods(train_x, train_y):
    """ given training, testing df, perform model fitting and prediction"""

    #     X_test, y_test,test = X_y_generator(test)

    models = {"Logistic Regression": LogisticRegression(),
              "QDA": QuadraticDiscriminantAnalysis(),
              "LDA": LinearDiscriminantAnalysis(),
              "Decission Tree Classification": DecisionTreeClassifier(criterion="gini", max_depth=depth),
              "Bagging": BaggingClassifier(n_estimators=29),
              "Ada Boost": AdaBoostClassifier(learning_rate=0.1 ** power),
              "Random Forest": RandomForestClassifier(n_estimators=estimator),
              "Gradient Boosting": GradientBoostingClassifier(n_estimators=10)}


    score_list = []
    model_list = []
    for algo in models.keys():
        model = models[algo]
        model.fit(train_x, train_y)

        #         score_list.append(1-model.score(X_test, y_test))
        model_list.append(algo)

        kfold = KFold(n_splits=n, shuffle=True)
        mis = 1 - abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy')))
        score_list.append(mis)

    print "Misclassification Rate by %s: %s" % (model_list[score_list.index(max(score_list))], max(score_list))
    print model_list
    print score_list


#####################  feature information  #########################

category_lst = ['site_name','posa_continent','user_location_country',
                'user_location_region','user_location_city',
                'channel','srch_destination_id','srch_destination_type_id',
                'hotel_continent','hotel_country','hotel_market',
                'hotel_cluster','new_srch_destination_id']

feature_col = ['site_name','posa_continent','user_location_country','user_location_region',
               'user_location_city','orig_destination_distance','is_mobile','is_package',
               'channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id',
               'srch_destination_type_id','is_booking','cnt','hotel_continent','hotel_country',
               'hotel_market','year','month','day','dow','stay_days','before_days','pca1','pca2','pca3','new_srch_destination_id'] # todo

selected_features = ['new_srch_destination_id','is_booking','before_days','pca1','pca2','pca3','cnt']

###################### model selection parameters #####################

depth = 17
neighbour = 5
n = 10
estimator = 26
power = 1

#######################################################################
DEST_FILE = "destinations.csv"
train = pd.read_csv("training_sub.0.1.csv")
test = pd.read_csv("test_sub.0.1.csv")

train_x, train_y = training_processor(train)
test_x, test_y = testing_processor(test)


# run_model_methods(train_X, train_Y)  # todo: run all for model selection

######################## parameter tuning #############################################

kfold = KFold(n_splits=10, shuffle=True)


""" Tree"""
crit = "gini"
acc = []
Depth = range(1,30)
for depth in Depth:
    classifier = DecisionTreeClassifier(criterion=crit,max_depth=depth)
    classifier.fit(train_x, train_y)
    acc.append(abs(np.mean(cross_val_score(classifier, train_x, train_y, cv=kfold,scoring='accuracy'))))

print "Accuracy rate by depth %s is %s" % (Depth[acc.index(max(acc))], max(acc))
plt.plot(Depth, acc)
plt.ylabel('Mean cv-accuracy')
plt.xlabel('Depth')
plt.title("Decision Tree")
plt.show()


""" Bagging """
acc = []
Estimators = range(2,30)
kfold = KFold(n_splits=10, shuffle=True)  # todo: shuffled
for estimator in Estimators:
    model =  BaggingClassifier(n_estimators=estimator)
    model.fit(train_x, train_y)
    acc.append(abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold,scoring='accuracy'))))
print "Accuracy rate by depth %s is %s" % (Estimators[acc.index(max(acc))], max(acc))

# prediction
plt.plot(Estimators, acc)
plt.ylabel('Mean cv-accuracy')
plt.xlabel('estimator')
plt.title("Bagging")
plt.show()


""" Ada Boost """
rng = np.random.RandomState(1)
acc = []
rates = range(1,10)
for rate in rates:
    classifier2 = AdaBoostClassifier(learning_rate=0.1 ** rate)
    classifier2.fit(train_x, train_y)
    acc.append(abs(np.mean(cross_val_score(classifier2, train_x, train_y, cv=kfold,scoring='accuracy'))))
print max(acc)
print rates[acc.index(max(acc))]

# prediction
plt.plot(rates, acc)
plt.ylabel('Mean cv-accuracy')
plt.xlabel('rates')
plt.title("Ada Boost")
plt.show()


""" Random Forest """
acc = []
Estimators = range(2,30)
kfold = KFold(n_splits=10, shuffle=True)  # todo: shuffled
for estimator in Estimators:
    model =  RandomForestClassifier(n_estimators=estimator)
    model.fit(train_x, train_y)
    acc.append(abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold,scoring='accuracy'))))
print "Accuracy rate by depth %s is %s" % (Estimators[acc.index(max(acc))], max(acc))

# prediction
plt.plot(Estimators, acc)
plt.ylabel('Mean cv-accuracy')
plt.xlabel('estimator')
plt.show()


""" Gradient Boosting """
acc = []
P = range(1,5)
kfold = KFold(n_splits=10, shuffle=True)  # todo: shuffled
for p in P:
    model =  GradientBoostingClassifier(learning_rate=0.1 ** p)
    model.fit(train_x, train_y)
    acc.append(abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold,scoring='accuracy'))))
print "Accuracy rate by depth %s is %s" % (P[acc.index(max(acc))], max(acc))

# prediction
plt.plot(P, acc)
plt.ylabel('Mean cv-accuracy')
plt.xlabel('learning rate power vs mean cv-accuracy(base 0.1)')
plt.show()