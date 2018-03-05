# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 08:42:50 2018

@author: CWang
"""
import pandas as pd

# Read in data
data = pd.read_csv("diabetic_data.csv")
id_map = pd.read_csv('IDs_mapping.csv')

admission_type = pd.read_csv('admission_type_mapping.csv')
admission_source = pd.read_csv('admission_source_mapping.csv')
discharge_type = pd.read_csv('discharge_mapping.csv')


# Clean Data - Change pseudonyms to unknown
def prep_ID(id_map):
    changecols = ['Not Available', ' Not Available', 'Not Mapped', ' Not Mapped', 'NULL', 'Unknown/Invalid']
    id_map['description'] = ['unknown' if ((x in changecols) | 
            (pd.isnull(x))) else x for x in id_map['description']]
    return id_map

admission_type = prep_ID(admission_type)
admission_source = prep_ID(admission_source)
discharge_type = prep_ID(discharge_type)

# Merge data sets with ID
data = pd.merge(data, admission_type, how='left', on = 'admission_type_id')
data = pd.merge(data, admission_source, how='left', on='admission_source_id')
data = pd.merge(data, discharge_type, how='left', on='discharge_disposition_id')

# Clean Data - Rename columns and replace ? with unknown
data = data.rename(index=str, columns={'description_x': 'admission_type', 
                                'description_y': 'admission_source',
                                'description': 'discharge_disposition'})
data.drop(['admission_type_id', 'admission_source_id', 'discharge_disposition_id'], axis=1, inplace=True)
data.replace('?', 'unknown', inplace=True)


# Clean Data - Replace codes with category names
def create_map():
    ## List of tuples containing name and range of codes (number of appearances of each)
    ## Infection is 1 to 139, so 139 occurences, Neoplasm is 100 occurences
    classify_list = [('Infection', 139),
                ('Neoplasm', (239 - 139)),
                ('Endocrine', (279 - 239)),
                ('Blood', (289 - 279)),
                ('Mental', (319 - 289)),
                ('Nervous', (359 - 319)),
                ('Sense', (389 - 359)),
                ('Circulatory', (459-389)),
                ('Respiratory', (519-459)),
                ('Digestive', (579 - 519)),
                ('Genitourinary', (629 - 579)),
                ('Pregnancy', (679 - 629)),
                ('Skin', (709 - 679)),
                ('Musculoskeletal', (739 - 709)),
                ('Congenital', (759 - 739)),
                ('Perinatal', (779 - 759)),
                ('Ill-defined', (799 - 779)),
                ('Injury', (999 - 799))]
    ## Loop over the tuples to create a dictionary to map codes to the names
    out_dict = {}
    count = 1
    for classification, num in classify_list:
        for i in range(num):
          out_dict.update({str(count): classification})  
          count += 1
    return out_dict
  

def map_codes(df, codes):
    col_names = df.columns.tolist() #list of column names
    for col in col_names:
        temp = [] 
        for num in df[col]: #check each entry in the column           
            if ((num is None) | (num in ['unknown', '?']) | (pd.isnull(num))): temp.append('unknown')   #if null, unknown, or ?, then make unknown
            elif(num.upper()[0] == 'V'): temp.append('supplemental')    #V is supplemental
            elif(num.upper()[0] == 'E'): temp.append('injury')          #E is injury
            else: 
                lkup = num.split('.')[0]    #if entry is multiple numbers separated by '.', take the first section and add it
                temp.append(codes[lkup])           
        df.loc[:, col] = temp               
    return df 

codes = create_map()
col_list = ['diag_1', 'diag_2', 'diag_3']   #Select diagnostic columns to change
data[col_list] = map_codes(data[col_list], codes)


# Clean Data - make two classes for readmitted column
def set_readmit(readmit_col):
    return ['NO' if (y == 'NO') else 'YES' for y in readmit_col]
data['readmitted'] = set_readmit(data['readmitted'])

# Separate into data and outcome
x = data.drop('readmitted', axis=1)
y = data['readmitted']


# Standardize Data
from sklearn import preprocessing
cols_to_stand = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
x[cols_to_stand] = preprocessing.scale(x[cols_to_stand])


# Make dummy variables (Just do all first time through, will edit later)
x = pd.get_dummies(x)


## Split the data into training and test sets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=123)

# Instantiate (or create) the specific machine learning model you want to use
lr = LogisticRegression()
lr.fit(x_train, y_train)    # Fit the model to the training data
predictions = lr.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('Logistic Regression accuracy: ' + str(accuracy))



## Cross-Validation
from sklearn.model_selection import cross_val_score
lr = LogisticRegression()
scores = cross_val_score(lr, x, y, cv=10)
accuracy = scores.mean()

print('Logistic Regression Scores: ' + str(scores))
print('Logistic Regression Accuracy: ' + str(accuracy))