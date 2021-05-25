#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

############# FUNCTIONS #############

###*** DATA processing ***###

# Loads the dataset and creates csv document for data analysis
def loadData():
    # Loading the data train set
    df = pd.read_csv('data/data_train.csv')
    # Creates document with the data analysis
    df.describe(include='all').to_csv('data/DataAnalysis.csv')
    # Displays info about data
    df.info(verbose=True, null_counts=True)
    # In both train and test set there was found one missing row - dropping NaN rows only for training,
    #because in instructions for leaderbaord the row with missing data was predicted
    df.dropna(inplace=True)
    return df

def dataInspection(df):
    # Checking correlation for color space data - it was found that they are copies of each other
    colorDF = df.loc[:, 'lightness_0_0':'blueyellow_2_2']
    print(colorDF.corr())
    # Checking labels and their representation
    print("Number of differetn colors: " + str(len(df['color'].unique())))
    print("Number of differetn textures: " + str(len(df['texture'].unique())))
    # Checking label classes representations and finding that the dataset is imbalanced
    print("Color class categories and number of their representations: ")
    print(df['color'].value_counts())
    print("Texture class categories and number of their representations: ")
    print(df['texture'].value_counts())

def dropDuplicates(df):
    count=0
    for i in df.duplicated():
        if i==True:
            count+=1
    print("Number of duplicates: " + str(count))
    df.drop_duplicates(inplace = True)
    return df

def cleanData(df):
    df = dropDuplicates(df)
    # Dropping some culumns that are obviously not relevat for prediction
    df.drop(['x', 'y', 'w', 'h', 'id', 'image'], inplace = True, axis=1)
    # Thanks to correlation it was discovered that lightness, redgreen and blueyellow contained the same information
    toDrop = df.loc[:, 'redgreen_0_0':'blueyellow_2_2'].columns
    df.drop(columns=toDrop, inplace=True)
    # After visual inspection - there are many 0s in hog data, it was deduced that these values are missing
    # The 0 values in the dataset are thus substituted with NaN value and further processed by the preprocessor pipeline
    #df.replace(to_replace=0, value = np.nan, inplace = True)
    return df

def make_hist(df):
    df.hist(bins=50, figsize=(100,85))
    plt.show()

    
###*** Training ***###

### Imbalanced data techniques ###

# 1. Under-sampling - Manually created function for custom undersampling
# It was experimented with this function, but did not bring any improvements
def underSample(df, value, num):
    # index and length of the rest of the values
    length_rest_class = len(df[df['color'] != value])
    index_rest = df[df['color'] != value].index
    # majority random choice delete to have the same length entered num parameter
    length_class= num
    index_majority_class = df[df['color'] == value].index
    index_random_maj = np.random.choice(index_majority_class, length_class, replace=False)
    # Get undersampling indexes
    under_sample_index = np.concatenate([index_random_maj, index_rest])
    # Creating new dataframe with the indexes
    under_sample = df.loc[under_sample_index]
    return under_sample

# 2. Using SMOTE technique for over-sampling 
def overSampleSMOTE(X,y, preprocesor):
    print('Shape before SMOTE')
    print(X.shape)
    print(y.shape)
    # Using SMOTE from imblearn
    smote = SMOTE(random_state = 0)
    preprocessor = preprocessorWithoutFeatSel()
    X = preprocessor.fit_transform(X)
    X, y = smote.fit_resample(X,y)
    print('Shape after SMOTE')
    print(X.shape)
    print(y.shape)
    return X,y

# 3. Balancing weights in the training model - it gave me the best result out of these 3 for SVM


### Preprocessing ###

# Sets up pipeline - imputes, scale data and also performs feature selection
def preprocessor():
    # There are no missing numeric values but in case some test sets have, it fills NaNs with mean values  
    preprocessor = Pipeline(steps=[('imputer_num', SimpleImputer(missing_values=np.nan, strategy='mean')), 
                                      ('scaler', StandardScaler()), 
                                      ('feature_selection', SelectFromModel(ExtraTreesClassifier(random_state=42)))])
    return preprocessor

# Preprocessor without feature selection
def preprocessorWithoutFeatSel():
    # There are no missing numeric values but in case some test sets have, it fills NaNs with mean values  
    preprocessor = Pipeline(steps=[('imputer_num', SimpleImputer(missing_values=np.nan, strategy='mean')), 
                                      ('scaler', StandardScaler())])
    return preprocessor

# Feature Selection separately
def featureSelection(X_train, y_train):
    print("Shape of X before feature selection: " + str(X_train.shape))
    preprocessor = preprocessorWithoutFeatSel()
    X_train = preprocessor.fit_transform(X_train)
    clf = ExtraTreesClassifier(random_state=42)
    clf = clf.fit(X_train, y_train)
    
    # Prints individual feature importances
    print(clf.feature_importances_)  

    selectModel = SelectFromModel(clf, prefit=True)
    X_train = selectModel.transform(X_train)
    print("Shape of X after feature selection: " + str(X_train.shape))
    return X_train


### SVM - model ###

# Function that trains SVM model for given class_label (color or texture)
def trainSVMmodel(df, class_label):
    
    # Splitting prediction label from the rest of the data
    y_train = df[class_label]
    X_train = df.drop(['color', 'texture'], axis=1)
    
    # Calls preprocessor
    preproces = preprocessor()

    if (class_label == 'color'):
        
        # Model with the best parameters for the color class
        model = SVC(class_weight='balanced', decision_function_shape='ovr', break_ties=True, C=1, gamma=0.01)
        
        # Setting up the final pipeline
        classifier = Pipeline(steps=[('preprocessor', preproces), ('classifier', model)])
    
    elif (class_label == 'texture'):
        
        # Model with the best parameters for the texture class
        model = SVC(class_weight='balanced', decision_function_shape='ovr', break_ties=True, C=1000000, gamma=1e-09)
        
        # Setting up the final pipeline
        classifier = Pipeline(steps=[('preprocessor', preproces), ('classifier', model)])
        
    else:
        print("Input color or texture class label")
        sys.exit()

    # Fitting into the pipeline which performs all the transformations and finally the model fitting
    classifier.fit(X_train, y_train)

    # Getting the predicted label
    y_train_hat = classifier.predict(X_train)
    
    # Here it saves trained model
    with open("models/SVM-model-" + class_label + ".pkl","wb") as f:
        pickle.dump(classifier,f)
    
    # Calling evaluation function
    TrainEvaluation(classifier, X_train, y_train, y_train_hat)
    
### Logisctic regression model ###

# Simple logistic regression classifier
def trainLogRegModel(df, class_label):
    
    # Splitting prediction label from the rest of the data
    y_train = df[class_label]
    X_train = df.drop(['color', 'texture'], axis=1)
    
    # Calls preprocessor
    preproces = preprocessor()

    if (class_label == 'color'):
        
        # Model with the best parameters for the color class
        model = LogisticRegression(class_weight = 'balanced', multi_class='ovr')
        
        # Setting up the final pipeline
        classifier = Pipeline(steps=[('preprocessor', preproces), ('classifier', model)])
    
    elif (class_label == 'texture'):
        
        # Model with the best parameters for the texture class
        model = LogisticRegression(class_weight = 'balanced', multi_class='ovr')
        
        # Setting up the final pipeline
        classifier = Pipeline(steps=[('preprocessor', preproces), ('classifier', model)])
        
    else:
        print("Input color or texture class label")
        sys.exit()

    # Fitting into the pipeline which performs all the transformations and finally the model fitting
    classifier.fit(X_train, y_train)

    # Getting the predicted label
    y_train_hat = classifier.predict(X_train)
    
    # Here it saves trained model
    with open("models/LogReg-model-" + class_label + ".pkl","wb") as f:
        pickle.dump(classifier,f)
    
    # Calling evaluation function
    TrainEvaluation(classifier, X_train, y_train, y_train_hat)
    
    
### Random Forest model ###

def trainRandForestModel(df, class_label):
    # Splitting prediction label from the rest of the data
    y_train = df[class_label]
    X_train = df.drop(['color', 'texture'], axis=1)
    
    # Calls preprocessor
    preproces = preprocessor()

    if (class_label == 'color'):
        
        # Model with the best parameters for the color class
        model = RandomForestClassifier(bootstrap = True, max_depth = 60,
                                       max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 1600)
        
        # Setting up the final pipeline
        classifier = Pipeline(steps=[('preprocessor', preproces), ('classifier', model)])
    
    elif (class_label == 'texture'):
        
        # Model with the best parameters for the texture class
        model = RandomForestClassifier(bootstrap = True, max_depth = 60, 
                                       max_features ='sqrt', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 1000)
        
        # Setting up the final pipeline
        classifier = Pipeline(steps=[('preprocessor', preproces), ('classifier', model)])
        
    else:
        print("Input color or texture class label")
        sys.exit()

    # Fitting into the pipeline which performs all the transformations and finally the model fitting
    classifier.fit(X_train, y_train)

    # Getting the predicted label
    y_train_hat = classifier.predict(X_train)
    
    # Here it saves trained model
    with open("models/RandomForest-model-" + class_label + ".pkl","wb") as f:
        pickle.dump(classifier,f)
    
    # Calling evaluation function
    TrainEvaluation(classifier, X_train, y_train, y_train_hat)

    
### Hyperparameters tunning ###

# Hyper parameter tunning - Grid serarch
# pridat tady i estimator do parametru
def hyperparameterTun(df, class_label, model_name):
    # Splitting prediction label from the rest of the data
    y_train = df[class_label]
    X_train = df.drop(['color', 'texture'], axis=1)
    
    preprocess = preprocessor()
    X_train_scaled = preprocess.fit_transform(X_train, y_train)

    if (model_name=='SVM'):
        # Different parameters to try
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)

        #C_range = np.linspace(0.01, 100, 20)
        #gamma_range = np.linspace(0.00001, 100, 20)

        param_grid = dict(gamma=gamma_range, C=C_range)

        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        grid = GridSearchCV(SVC(class_weight='balanced', decision_function_shape='ovr', break_ties=True),
                        scoring='balanced_accuracy', param_grid=param_grid, cv=cv)
        grid.fit(X_train_scaled, y_train)

        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
        
    elif (model_name=='RandForest'):
        
        # Different parameters to try
        param_grid = {'bootstrap': [True], 'max_depth': [20, 30, 40, 60],
         'max_features': ['sqrt'], 'min_samples_leaf': [2, 4], 'min_samples_split': [5, 10], 
         'n_estimators': [600, 1000, 1400, 1600]}

        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        grid = GridSearchCV(RandomForestClassifier(),
                        scoring='balanced_accuracy', param_grid=param_grid, cv=cv)
        grid.fit(X_train_scaled, y_train)

        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))


    
###*** Evaluation ***###

def TrainEvaluation(classifier, X_train, y_train, y_train_hat):

    # Training accuracy score
    print("Accuracy on training set: " + str(accuracy_score(y_train, y_train_hat)))

    # Training balanced accuracy score
    print("Balanced accuracy on training set: " + str(balanced_accuracy_score(y_train, y_train_hat)))

    # Classification report
    print(classification_report(y_train, y_train_hat))
    plot_confusion_matrix(classifier, X_train, y_train)

    # Cross-validation accuracy on train set
    stratifiedFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(classifier, X_train, y_train, cv=stratifiedFold, scoring=['balanced_accuracy','accuracy'])
    print("Accuracy score on cross validation: " + str(scores['test_accuracy']))
    print("Balanced accuracy score on cross validation: " + str(scores['test_balanced_accuracy']))

    
###*** Predicting on test set ***###

def testPredictions(model_name):
    df = pd.read_csv('data/data_test.csv')
    df = cleanData(df)
    X_test = df
    
    if(model_name=='SVM'):
        # Loads color SVM color model
        try:
            with open("models/SVM-model-color.pkl","rb") as f:
                SVM_model_color = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()

        # Loads texture SVM texture model
        try:
            with open("models/SVM-model-texture.pkl","rb") as f:
                SVM_model_texture = pickle.load(f)
        except FileNotFoundError:
            print("Train the model first")
            sys.exit()
            
        # Predicts
        predictTest(SVM_model_color, X_test, 'color', model_name)
        predictTest(SVM_model_texture, X_test, 'texture', model_name)
        
    else:
        print("Input corrrect trained model")
        sys.exit()

# Executes the prediction and creates txt file with predictions
def predictTest(trained_model, X_test, class_label, model_name):
    y_test_hat = trained_model.predict(X_test)
    df_y_test_hat = pd.DataFrame(y_test_hat)
    df_y_test_hat.to_csv('test_results/y_test_' + model_name + '_' + class_label + '.csv', header=False, index=False)


    


# In[109]:


############# Main program #############

color_class_label = 'color'
texture_class_label = 'texture'
SVM_model_label = 'SVM'
RandForest_model_label = 'RandForest'


#** Preprocessing **#

# loads data
df = loadData()
# inspects data
dataInspection(df)
# Clean data according to findings while analysing them
df = cleanData(df)


#** SVM Training **#

# Color
trainSVMmodel(df, color_class_label)

# Texture
trainSVMmodel(df, texture_class_label)


#** SVM hyperparameters tunning **#

# Color
hyperparameterTun(df,color_class_label, SVM_model_label)
# The best parameters are {'C': 1.0, 'gamma': 0.01} with a score of 0.19
# Achieving 20.52 % accuracy on test set

# Texture
hyperparameterTun(df,texture_class_label, SVM_model_label)
# The best parameters are {'C': 1000000.0, 'gamma': 1e-09} with a score of 0.33
# Achieving 41.324 % accuracy on test set


#** SVM Test set predictions **#
testPredictions('SVM')


#** Logistic Regression Training **#
trainLogRegModel(df, color_class_label)
trainLogRegModel(df, texture_class_label)


#** Random Forest Training **#
trainRandForestModel(df, color_class_label)
trainRandForestModel(df, texture_class_label)


# Hyper tunning
hyperparameterTun(df,color_class_label, RandForest_model_label)
#The best parameters for color are {'bootstrap': True, 'max_depth': 60,
#                         'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 
#                         'n_estimators': 1600} with a score of 0.17
hyperparameterTun(df,texture_class_label, RandForest_model_label)
#The best parameters for texture are {'bootstrap': True, 'max_depth': 60,
#                         'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 
#                         'n_estimators': 1000} with a score of 0.20
#

