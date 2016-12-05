import pandas as pd
import random
import ml_metrics as metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

########################################################################################################################
# Read in training dataset
########################################################################################################################
# Read in training dataset
train0 = pd.read_csv('training_2014_0.1sub.csv')
category_lst = ['site_name','posa_continent','user_location_country',
                'user_location_region','user_location_city',
                'channel','srch_destination_id','srch_destination_type_id',
                'hotel_continent','hotel_country','hotel_market',
                'hotel_cluster']

random_rows = random.sample(train0.index, 10000)
train0 = train0[train0['is_booking'] == 1]
train0 = train0.dropna(axis=0, how='any')

train_random = train0.ix[random_rows]
train, test = train_test_split(train_random, test_size=0.2, random_state=42)
train = train.dropna(axis=0, how='any')
test = test.dropna(axis=0, how='any')

for col in category_lst:
    train[col] = train[col].astype('category')

# Read in latent destinaton variables
destinations = pd.read_csv("destinations.csv")
pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small.columns = ['pca1', 'pca2', 'pca3']
dest_small["srch_destination_id"] = destinations["srch_destination_id"]

# -generating features
# ADD NEW FEATURES
# --feature 1 & 2: stay duration and how long before check-in
def delta_to_int(timedelta):
    try:
        return timedelta.days
    except AttributeError:
        return np.nan


def date_to_weekday(date):
    return date.weekday()

feature_col = ['site_name','posa_continent','user_location_country','user_location_region',
       'user_location_city','orig_destination_distance','is_mobile','is_package',
       'channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id',
       'srch_destination_type_id','is_booking','cnt','hotel_continent','hotel_country',
       'hotel_market','year','month','day','dow','stay_days','before_days','pca1','pca2','pca3']

selected_features = ['cnt','hotel_country','hotel_continent','pca2','srch_rm_cnt','before_days','site_name']


def calc_fast_features(df):
    df['dow'] = pd.to_datetime(df['date_time']).apply(date_to_weekday)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['srch_ci'] = pd.to_datetime(df['srch_ci'], errors='coerce')
    df['srch_co'] = pd.to_datetime(df['srch_co'], errors='coerce')
    df['stay_days'] = (df['srch_co'] - df['srch_ci'])
    df['before_days'] = (df['srch_ci'] - df['date_time'])
    df['stay_days'] = df['stay_days'].apply(delta_to_int)
    df['before_days'] = df['before_days'].apply(delta_to_int)
    df1 = df.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    df2 = df1.drop("srch_destination_iddest", axis=1)
    #df3 = df2[feature_col]
    df3 = df2[selected_features]
    return df3


train_X = calc_fast_features(train)
train_Y = train['hotel_cluster']

imp = Imputer(strategy='mean', axis=0)
train_X1 = imp.fit_transform(train_X)

clf = RandomForestClassifier(n_estimators=20, min_weight_fraction_leaf=0.1)
scores = cross_val_score(clf, train_X1, train_Y, cv=10)
print 'Random Forest Regression accuracy score:', np.abs(scores.mean())

########################################################################################################################
# Feature Selection
########################################################################################################################
# clf = clf.fit(train_X1, train_Y)
# importances = clf.feature_importances_
#
# std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# print("Feature ranking:")
#
# for f in range(train_X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
# for i in range(len(feature_col)):
#     print 'feature ', i+1, ':', feature_col[i]

# ########################################################################################################################
# # Predict the value using binary
# ########################################################################################################################
imp = Imputer(strategy='mean', axis=0)
test_X = calc_fast_features(test)
test_X1 = imp.fit_transform(test_X)
test_Y = test['hotel_cluster']

train_Y_dummy = pd.get_dummies(train_Y)

all_probs = []
for col in train_Y_dummy.columns:
    Y_train = train_Y_dummy[col]
    clf = RandomForestClassifier()
    clf.fit(train_X1, Y_train)
    probs = []
    preds = clf.predict_proba(test_X1)
    probs.append([p[1] for p in preds])
    all_probs.append(probs[0])

prediction_frame = pd.DataFrame(all_probs).T


def find_top_5(row):
    return list(row.nlargest(5).index)

preds = []
for index, row in prediction_frame.iterrows():
    preds.append(find_top_5(row))

print test_Y.tolist()
print metrics.mapk([[l] for l in test_Y], preds, k=100)