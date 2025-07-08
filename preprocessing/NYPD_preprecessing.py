import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the comma delimited data file
path = "/home/kxj200023/data/NYPD/2011.csv"
initial_data = pd.read_csv(path, encoding='latin-1', low_memory=False)

# Check for presence of duplicate rows before filtering to relevant features
initial_data.head()

# Extract relevant features (predictor and response) that will be utilized for analysis and remove those that will result in data leakage
# Precint (pct) is retained to clean city feature, but will be removed later on due to difficulty in binning them into separate pricints

# snf_data_pre = 'stop and frisk' data before preliminary processing
# Predictor features include description of suspect, officer behaviour and circumstances of the stop
# Response feature is whether the suspect was frisked
snf_data_pre = initial_data[['sex','race','age','ht_feet','ht_inch','weight','haircolr','eyecolor','build',
                            'city','pct','timestop','inout','trhsloc','typeofid','othpers',
                            'explnstp','offunif','officrid','offverb','offshld',
                            'ac_rept','ac_proxm','ac_evasv','ac_assoc','ac_cgdir','ac_incid','ac_time','ac_stsnd',
                            'frisked']].copy()

# Numeric features
x = snf_data_pre.describe().transpose()

### Calculate total height of individuals in inches by combining ht_feet and ht_inch

# Create new column combining feet and inches into one number
snf_data_pre['ht_inch'] = (snf_data_pre['ht_feet'] * 12) + snf_data_pre['ht_inch']

# Delete feet column
del snf_data_pre['ht_feet']

# Rename ht_inch to height
snf_data_pre.rename(columns={'ht_inch': 'height'}, inplace=True)

snf_data_pre.describe().transpose()

# Get data types
snf_data_pre.dtypes

# Get list of column names which are of type string (categorical)
str_cols = snf_data_pre.select_dtypes(['object']).columns

# String features
snf_data_pre[str_cols].describe()
# There are a number of features with missing values (city, officrid, offverb, offshld)

# Create deep copy of dataframe in case there's need to revert to original dataframe
snf_data_toclean = snf_data_pre.copy()
x = snf_data_toclean.shape

# Replace empty values of parameters with Nan
snf_data_toclean.replace(to_replace=' ', value=np.NaN, inplace=True)

# Count number of empty values
snf_data_toclean.isnull().sum()

# View pct (precints) of missing city data
snf_data_toclean.loc[snf_data_toclean['city'].isnull()]['pct'].unique()

# Impute missing city values using precint data

for i in [1, 5, 6, 7, 9, 10, 13, 14, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28, 30, 32, 33, 34]:
  snf_data_toclean.loc[snf_data_toclean['pct']==i, 'city']='MANHATTAN'

for i in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52]:
  snf_data_toclean.loc[snf_data_toclean['pct']==i, 'city']='BRONX'

for i in [60,61,62,63,66,67,68,69,70,71,72,73,75,76,77,78,79,81,83,84,88,90,94]:
  snf_data_toclean.loc[snf_data_toclean['pct']==i, 'city']='BROOKLYN'

for i in [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]:
  snf_data_toclean.loc[snf_data_toclean['pct']==i, 'city']='QUEENS'

for i in [120,121,122,123]:
  snf_data_toclean.loc[snf_data_toclean['pct']==i, 'city']='STATEN IS'

# Check all city values have been imputed correctly (no null values)
x = snf_data_toclean.isnull().sum()['city']

# Drop precint column as no longer needed
snf_data_toclean.drop(labels='pct', axis=1, inplace=True)

# View quantity of unique values in officrid, offverb and offshld
# print(snf_data_toclean['officrid'].value_counts())
# print(snf_data_toclean['offverb'].value_counts())
# print(snf_data_toclean['offshld'].value_counts())

# Print shape of dataframe before dropping columns
# print(snf_data_toclean.shape)
# Drop the 3 columns
snf_data_toclean.drop(labels=['officrid', 'offverb', 'offshld'], axis=1, inplace=True)

# Get updated list of column names which are of type string (categorical) and integer
str_cols = snf_data_toclean.select_dtypes(['object']).columns
int_cols = snf_data_toclean.select_dtypes(['int64']).columns

# View unique values for each column
# for i in str_cols:
#   print (snf_data_toclean[i].value_counts())
#   print()

# Replace all 'other' and 'unknown' values with NaN and drop those rows
snf_data_toclean = snf_data_toclean.replace(['Z', 'ZZ', 'XX'], np.NaN)
snf_data_toclean.dropna(inplace=True)

# Check if rows with NaN were removed
snf_data_toclean[snf_data_toclean.isna().any(axis=1)].shape

# Remove rows with value 'O' in typeofid column
snf_data_toclean = snf_data_toclean[snf_data_toclean['typeofid']!='O']

# Check if rows removed
snf_data_toclean[snf_data_toclean['typeofid']=='O'].shape

# Remove rows with value 'U' in race column
snf_data_toclean = snf_data_toclean[snf_data_toclean['race']!='U']

# Check if rows removed
snf_data_toclean[snf_data_toclean['race']=='U'].shape

# Replace erroneous value on haircolor column
snf_data_toclean['haircolr'] = snf_data_toclean['haircolr'].replace({'RA': 'RD'})
snf_data_toclean['haircolr'].value_counts()

# Replace erroneous value on eyecolor column
snf_data_toclean['eyecolor'] = snf_data_toclean['eyecolor'].replace({'MC':'MA', 'P':'PK'})
snf_data_toclean['eyecolor'].value_counts()

# Check unique values for each column
# for i in str_cols:
#   print (snf_data_toclean[i].value_counts())
#   print()

# Create deep copy of processed dataset
snf_data_feat = snf_data_toclean.copy()
x = snf_data_feat.shape

### Remove erroneous ages, heights and weights

# Remove rows with ages below 5 and above 100 years old
snf_data_feat = snf_data_feat[(snf_data_feat['age']>=5) & (snf_data_feat['age']<=100)]

# Remove rows with heights below 40 in and above 90 in
snf_data_feat = snf_data_feat[(snf_data_feat['height']>=40) & (snf_data_feat['height']<=90)]

# Remove rows with weights below 50 lbs and above 300 lbs
snf_data_feat = snf_data_feat[(snf_data_feat['weight']>=50) & (snf_data_feat['weight']<=300)]

# Check if some rows were removed correctly
x = snf_data_feat[['age','height','weight']].describe().transpose()


# Visualize distribution of ages, heights and weights
# plt.subplot(131)
# snf_data_feat.boxplot('age')
# plt.subplot(132)
# snf_data_feat.boxplot('height')
# plt.subplot(133)
# snf_data_feat.boxplot('weight')
# plt.subplots_adjust(left=0.0, bottom=0.0, right=2.5, top=1, wspace=0.2, hspace=0.2)

# View number of unique values for each cateogorical parameter
x = snf_data_feat[str_cols].nunique()


### sex parameter
# View unique values for sex parameter
# print(snf_data_feat['sex'].value_counts())

# Convert sex parameter into binary
snf_data_feat['sex'] = snf_data_feat['sex'].replace({'M': 1, 'F': 0})

# Check if values converted correctly
# print(snf_data_feat['sex'].value_counts())

### inout parameter
# print(snf_data_feat['inout'].value_counts())

# Convert inout parameter into binary
snf_data_feat['inout'] = snf_data_feat['inout'].replace({'I': 1, 'O': 0})

# Check if values converted correctly
# print(snf_data_feat['inout'].value_counts())

# Retrieve list of parameters with yes/no values
yesno_cols = str_cols[snf_data_feat[str_cols].nunique() == 2][2:]

### Yes/No parameter conversion
# View unique values for different yes/no parameters
snf_data_feat[yesno_cols].apply(pd.value_counts)

# Convert Yes/No columns to integers (1/0)
snf_data_feat[yesno_cols] = snf_data_feat[yesno_cols].replace({'N':0, 'Y':1})

# Check if values converted correctly
snf_data_feat[yesno_cols].apply(pd.value_counts)

# Retrieve list of parameters with more than 2 values
ohc_cols = str_cols[snf_data_feat[str_cols].nunique() > 2]

 # Convert all categorical data into binary using One Hot Encoding
ohc_df = pd.get_dummies(snf_data_feat[ohc_cols])

# Concatenate One Hot Encoding dataframe with original dataframe and the label (frisked) as the last column
snf_data_feat = pd.concat([snf_data_feat.iloc[:,:-1], ohc_df, snf_data_feat.iloc[:,-1]], axis=1)

# Remove columns
snf_data_feat.drop(ohc_cols, axis=1, inplace=True)

snf_data_feat.head()

# Get list of columns which have binary values
binary_cols = snf_data_feat.columns[5:].insert(loc=0, item=snf_data_feat.columns[0])

# View number of values
snf_data_feat[binary_cols].apply(pd.value_counts)

snf_data_feat.describe().iloc[:3]

# from sklearn.ensemble import RandomForestClassifier
# from sklearn import preprocessing
# import warnings                           # silence warnings that commonly occur with random forest
# warnings.filterwarnings('ignore')
#
# # Create deep copy for feature selection
# snf_data_feat_selec = snf_data_feat.copy()
#
# # Perform feature selection using random forest classifier
# x = snf_data_feat_selec.iloc[:,:-1]
# y = snf_data_feat_selec.iloc[:,-1]
#
# lab_enc = preprocessing.LabelEncoder(); y_encoded = lab_enc.fit_transform(y) # this removes an encoding error
#
# random_forest_feat = RandomForestClassifier(random_state = 50)   # instantiate the random forest
# random_forest_feat = random_forest_feat.fit(x,np.ravel(y_encoded)) # fit the random forest
# importances = random_forest_feat.feature_importances_ # extract the expected feature importances
# # std = np.std([tree.feature_importances_ for tree in random_forest_feat.estimators_],axis=0) # calculate stdev over trees
# feat_ranks = np.argsort(importances)[::-1]   # find feature ranks in descending order
#
# # Print feature ranking
# print("Feature Ranking (The feature importance values sum to 1):")
# for f in range(x.shape[1]):
#     print("%d. %s (%.4f)" % (f + 1, x.columns.values[feat_ranks[f]], importances[feat_ranks[f]]))
#
# # Create deep copy of dataframe before filtering
# snf_data_filtered = snf_data_feat.copy()
#
# # Filter out features to the top 10 most important ones
# feat_to_filter = feat_ranks[10:]
# print("Number of features to remove:", feat_to_filter.size)
#
# # Drop filtered features
# # snf_data_filtered.drop(snf_data_feat.columns[feat_to_filter], axis = 1, inplace = True)
#
# # Check if columns removed
# x = snf_data_filtered.head()

# Sample dataset using test train split to keep records that were not sampled
snf_data_analysis, snf_data_unused = train_test_split(snf_data_feat, train_size=100000, random_state=100)

# Display shapes of dataset
print("Analysis dataset =",snf_data_analysis.shape)
print("Ununsed dataset =",snf_data_unused.shape)

snf_data_analysis.to_csv('/home/kxj200023/data/NYPD/data.csv')