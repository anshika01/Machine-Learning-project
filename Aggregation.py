import pandas as pd
import numpy as np


def most_relevant(group):
    relevance = group['relevance'].values
    hotel_cluster = group['hotel_cluster'].values
    relevance_index=np.argsort(relevance)[::-1]
    relevant = hotel_cluster[relevance_index][:5]
    return relevant

train = pd.read_csv("train2014.csv")
test=pd.read_csv("test2014.csv")

click_rel=0.30

#'Count' - total number of times the combination appeared together
#'Sum' - total number of time the booking was done with his combination

grp_agg = train.groupby(['srch_destination_id','hotel_cluster'])['is_booking'].agg(['sum','count'])
grp_agg.reset_index(inplace=True)
#grp_agg = grp_agg.groupby(['srch_destination_id','hotel_cluster']).sum().reset_index()
grp_agg['count'] = grp_agg['count']-grp_agg['sum']

#count reduced by frequency of booking will give number of clicks
grp_agg = grp_agg.rename(columns={'sum':'bookings','count':'clicks'},inplace=True)

#used Cross validation to find best estimate of click weight, which can be approximated to 0.30
grp_agg['relevance'] = grp_agg['bookings'] + click_rel * grp_agg['clicks']
most_rel = grp_agg.groupby(['srch_destination_id']).apply(most_relevant)
most_rel = pd.DataFrame(most_rel).rename(columns={0:'hotel_cluster'},inplace=True)

test = test.merge(most_rel, how='left',left_on=['srch_destination_id'],right_index=True)

#converting the prediction for MAPK input formats.
##MAPK requires input to be list of lists.
preds=[]
for index, row in test.iterrows():
    preds.append(row['hotel_cluster_y'])
target = [[l] for l in test["hotel_cluster_x"]]

print "MAPK accuracy is", metrics.mapk(target, preds, k=5)
