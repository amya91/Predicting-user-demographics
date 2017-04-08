import pandas as pd
import numpy
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from keras.constraints import maxnorm

datadir = '../Data'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),index_col='device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

events = pd.read_csv(os.path.join(datadir,'events.csv'),parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'),usecols=['event_id','app_id','is_active'],dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

gatrain['trainrow'] = numpy.arange(gatrain.shape[0])

# feature engineering

# phone brand

# encoding
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']

# For phone brand data the data array will be all ones, row_ind will be the row number of a device 
# and col_ind will be the number of brand

Xtr_brand = csr_matrix((numpy.ones(gatrain.shape[0]),(gatrain.trainrow, gatrain.brand)))

# device model
# concatenating phone brand and device model
m = phone.phone_brand.str.cat(phone.device_model)

# encoding
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']

# For device model data the data array will be all ones, row_ind will be the row number of a device 
# and col_ind will be the number of brand-device model

Xtr_model = csr_matrix((numpy.ones(gatrain.shape[0]),(gatrain.trainrow, gatrain.model)))

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
                       .reset_index())

# building a feature matrix where the data is all ones, row_ind comes from trainrow or testrow 
# and col_ind is the label-encoded app_id
d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((numpy.ones(d.shape[0]), (d.trainrow, d.app)),shape=(gatrain.shape[0],napps))

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
                .reset_index())
                
# building a feature matrix where the data is all ones, row_ind comes from trainrow or testrow 
# and col_ind is the encoded label id
d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((numpy.ones(d.shape[0]), (d.trainrow, d.label)),shape=(gatrain.shape[0],nlabels))

# concatenate all features
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')

# deleting objects that are no longer needed
del phone
del appevents
del applabels
del Xtr_brand
del Xtr_model
del Xtr_app
del Xtr_label

MIN_VAL_ALLOWED = 15

nsp = numpy.squeeze(numpy.asarray(Xtrain.sum(axis=0) >= MIN_VAL_ALLOWED))
Xtrain = Xtrain[:,nsp].toarray()

# encoding demographic group
targetencoder = LabelEncoder().fit(gatrain.group)
encoded_y = targetencoder.transform(gatrain.group)
dummy_y = np_utils.to_categorical(encoded_y)
nclasses = len(targetencoder.classes_)

# model
model = Sequential()
model.add(Dense(Xtrain.shape[1], input_dim=Xtrain.shape[1], init = 'glorot_normal', activation='softsign', W_constraint=maxnorm(2)))
model.add(Dropout(0.6))
model.add(Dense(Xtrain.shape[1]*0.6, init = 'glorot_normal', activation='softsign', W_constraint=maxnorm(2)))
model.add(Dropout(0.6))
model.add(Dense(nclasses, init = 'glorot_normal', activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

model.fit(Xtrain, dummy_y, nb_epoch=20, batch_size=1000)

# evaluate the model
scores = model.evaluate(Xtrain, dummy_y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(Xtrain)

pred = pd.DataFrame(data = predictions,
                    columns = list(targetencoder.inverse_transform(list(set(encoded_y)))))

# creating a column to indicate which is our predicted group
pred['prediction'] = pred.idxmax(axis=1)

# writing to csv
pred.to_csv('output_ann.csv',index=True)

# reading back the ann and dropping the unnecessary columns
predictions = pd.read_csv('output_ann.csv')
del predictions['Unnamed: 0']

# merging with the test data to see if we've made the right predictions
gatrain['device id'] = gatrain.index
gatrain = gatrain.set_index('trainrow')
eva = gatrain.merge(predictions, left_index=True, right_index=True)
eva = eva.set_index('device id')
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
final.to_csv('../Visualization/datafiles/ANN.csv', index=False)