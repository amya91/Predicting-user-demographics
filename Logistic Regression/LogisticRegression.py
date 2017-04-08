import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack


datadir = '../Data'
data = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),index_col='device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

events = pd.read_csv(os.path.join(datadir,'events.csv'),parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'),usecols=['event_id','app_id','is_active'],dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

gatrain, gatest, y_train, y_test = train_test_split(data, data['group'], test_size=0.33, random_state=42)

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# feature engineering

# phone brand

# encoding
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']

# For phone brand data the data array will be all ones, row_ind will be the row number of a device 
# and col_ind will be the number of brand

Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]),(gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]),(gatest.testrow, gatest.brand)))

# device model
# concatenating phone brand and device model
m = phone.phone_brand.str.cat(phone.device_model)

# encoding
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']

# For device model data the data array will be all ones, row_ind will be the row number of a device 
# and col_ind will be the number of brand-device model

Xtr_model = csr_matrix((np.ones(gatrain.shape[0]),(gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]),(gatest.testrow, gatest.model)))

# active apps
# merge device_id column from events table to app_events
# group the resulting dataframe by device_id and app and aggregate
# merge in trainrow and testrow columns to know at which row to put each device in the features matrix

# encoding apps
appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)

napps = len(appencoder.classes_)

# merging tables
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())

# building a feature matrix where the data is all ones, row_ind comes from trainrow or testrow 
# and col_ind is the label-encoded app_id
d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)),shape=(gatrain.shape[0],napps))

d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)),shape=(gatest.shape[0],napps))

# app labels
# constructed in a way similar to apps features by merging app_labels with the 
# deviceapps dataframe we created above
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]

# encoding apps in app labels
applabels['app'] = appencoder.transform(applabels.app_id)

# encoding app labels
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)

# merging
devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
                
# building a feature matrix where the data is all ones, row_ind comes from trainrow or testrow 
# and col_ind is the encoded label id
d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)),shape=(gatrain.shape[0],nlabels))

d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)),shape=(gatest.shape[0],nlabels))

# concatenate all features
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')
fulltest = vstack((Xtrain, Xtest))

# encoding demographic group
#targetencoder = LabelEncoder().fit(gatrain.group)
targetencoder = LabelEncoder().fit(y_train)
y = targetencoder.transform(y_train)
nclasses = len(targetencoder.classes_)

# In order to make a good logistic regression model we need to choose a value for 
# regularization constant C. Smaller values of C mean stronger regularization and 
# its default value is 1.0. We probably have a lot of mostly useless columns 
# (rare brands, models or apps), so we'd better look at stronger regularization than default

# predict on test data
clf = LogisticRegression(C=0.02,multi_class='multinomial',solver='lbfgs')
clf.fit(Xtrain, y)
pred = pd.DataFrame(clf.predict_proba(fulltest), index = data.index, columns=targetencoder.classes_)

# creating a column to indicate which is our predicted group
pred['prediction'] = pred.idxmax(axis=1)

# merging with the test data to see if we've made the right predictions
eva = data.merge(pred, left_index=True, right_index=True)
eva = eva[['gender','group','prediction']]
eva['predicted gender'] = eva['prediction'].str[0]

# comparing actual and predicted gender and group
eva['gender result'] = (eva['gender'] == eva['predicted gender']).astype(int)
eva['group result'] = (eva['group'] == eva['prediction']).astype(int)

# dropping columns that are not needed
eva = eva[['prediction','predicted gender','gender result','group result']]

# changing column names
eva.columns = ['age','gender','gender result','age result']

# merging with events table to get latitude and longitude values
events = events.set_index('device_id')
# first delete all entries in events where lat long is zero or long<70
events = events[(events.latitude != 0) & (events.longitude >70)]

# then merge and drop duplicate entries
final = eva.merge(events, left_index=True, right_index=True, how='inner')
final['device id'] = final.index
final = final.drop_duplicates('device id',keep='first')

# dropping unnecessary columns
final = final[['device id','gender','age','latitude','longitude']]

# writing result
final.to_csv('../Visualization/datafiles/LogisticRegression.csv', index=False)