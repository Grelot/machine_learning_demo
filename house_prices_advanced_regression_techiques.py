#===============================================================================
#INFORMATION
#===============================================================================
# Codes for Predicting Housing Prices:
# A model to predict the value of a given house in the Boston real estate market
# using various statistical analysis tools.
# Identified the best price that a client can sell their house 
# using machine learning.
# 
# Guerin Pierre-Edouard
#
#
# git repository : https://github.com/Grelot/machine_learning_demo
#
#==============================================================================
#NOTICE
#==============================================================================
#
#
#
#
#==============================================================================
#MODULES
#==============================================================================
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, LabelEncoder
import scipy.stats as stats
import xgboost as xgb
import lightgbm as lgb
import warnings


warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')

#==============================================================================
#CLASS
#==============================================================================



#==============================================================================
#FUNCTIONS
#==============================================================================

def multiplot(data,features,plottype,nrows,ncols,figsize,y=None,colorize=False):
    """ This function draw a multi plot for 3 types of plots ["regplot","distplot","coutplot"]"""
    n = 0
    plt.figure(1)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)    
    if colorize:
        colors = sns.color_palette(n_colors=(nrows*ncols))
    else :
        colors = [None]*(nrows*ncols)        
    for row in range(nrows):
        for col in range(ncols):            
            if plottype == 'regplot':
                if y == None:
                    raise ValueError('y value is needed with regplot type')                
                sns.regplot(data = data, x = features[n], y = y ,ax=axes[row,col], color = colors[n])
                correlation = np.corrcoef(data[features[n]],data[y])[0,1]
                axes[row,col].set_title("Correlation {:.2f}".format(correlation))            
            elif plottype == 'distplot':
                sns.distplot(a = data[features[n]],ax = axes[row,col],color=colors[n])
                skewness = data[features[n]].skew()
                axes[row,col].legend(["Skew : {:.2f}".format(skewness)])            
            elif plottype in ['countplot']:
                g = sns.countplot(x = data[features[n]], y = y, ax = axes[row,col],color = colors[n])
                g = plt.setp(g.get_xticklabels(), rotation=45)                
            n += 1
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()

#==============================================================================
#MAIN
#==============================================================================


## download data https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

trainDatFile="data/house_prices_advanced_regression_techniques/train.csv"
testDatFile="data/house_prices_advanced_regression_techniques/test.csv"

train = pd.read_csv(trainDatFile)
test = pd.read_csv(testDatFile)

## number of descripteurs
train.shape
## variables quantitatives
train.select_dtypes(include=['int64','float64']).columns
## drop useless "Id" feature
train = train.drop(labels = ["Id"],axis = 1)
test = test.drop(labels = ["Id"],axis = 1)
## Let's explore the data.
g = sns.jointplot(x = train['GrLivArea'], y = train['SalePrice'],kind="reg")
g.annotate(stats.pearsonr)
## explatory analysis sale price
train['SalePrice'].describe()
g = sns.distplot(train['SalePrice'],color="gray")
g = g.legend(["Coefficient Assymetrie : {:.2f}".format(train['SalePrice'].skew())],loc='best')
## correlation
corrmat = train.corr()
g = sns.heatmap(train.corr())
## most correlated features
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
## regplot most correlated features
feats = ["GrLivArea","TotalBsmtSF", "YearBuilt", "1stFlrSF", "GarageCars", "GarageArea"]
multiplot(data = train, features = feats, plottype = "regplot",nrows = 3, ncols = 2, figsize = (10,6), y = "SalePrice", colorize = True)
## Join train and test datasets in order to avoid obtain the same number of feature during categorical conversion
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
## Infos
dataset.info()
## missing data
dataset = dataset.fillna(np.nan)
missing_features = dataset.columns[dataset.isnull().any()]
dataset[missing_features].isnull().sum().sort_values(ascending=False)
## replace NaN values by "No" for categorical variables
dataset["Alley"] = dataset["Alley"].fillna("No")
dataset["MiscFeature"] = dataset["MiscFeature"].fillna("No")
dataset["Fence"] = dataset["Fence"].fillna("No")
dataset["PoolQC"] = dataset["PoolQC"].fillna("No")
dataset["FireplaceQu"] = dataset["FireplaceQu"].fillna("No")
dataset["BsmtCond"] = dataset["BsmtCond"].fillna("No")
dataset["BsmtQual"] = dataset["BsmtQual"].fillna("No")
dataset["BsmtFinType2"] = dataset["BsmtFinType2"].fillna("No")
dataset["BsmtFinType1"] = dataset["BsmtFinType1"].fillna("No")
dataset.loc[dataset["BsmtCond"] == "No","BsmtUnfSF"] = 0
dataset.loc[dataset["BsmtFinType1"] == "No","BsmtFinSF1"] = 0
dataset.loc[dataset["BsmtFinType2"] == "No","BsmtFinSF2"] = 0
dataset.loc[dataset["BsmtQual"] == "No","TotalBsmtSF"] = 0
dataset.loc[dataset["BsmtCond"] == "No","BsmtHalfBath"] = 0
dataset.loc[dataset["BsmtCond"] == "No","BsmtFullBath"] = 0
dataset["BsmtExposure"] = dataset["BsmtExposure"].fillna("No")
dataset["Utilities"] = dataset["Utilities"].fillna("AllPub")
dataset["GarageType"] = dataset["GarageType"].fillna("No")
dataset["GarageFinish"] = dataset["GarageFinish"].fillna("No")
dataset["GarageQual"] = dataset["GarageQual"].fillna("No")
dataset["GarageCond"] = dataset["GarageCond"].fillna("No")
dataset["MasVnrType"] = dataset["MasVnrType"].fillna("None")
dataset.loc[dataset["MasVnrType"] == "None","MasVnrArea"] = 0
dataset.loc[dataset["GarageType"] == "No","GarageYrBlt"] = dataset["YearBuilt"][dataset["GarageType"]=="No"]
dataset.loc[dataset["GarageType"] == "No","GarageCars"] = 0
dataset.loc[dataset["GarageType"] == "No","GarageArea"] = 0
dataset["GarageArea"] = dataset["GarageArea"].fillna(dataset["GarageArea"].median())
dataset["GarageCars"] = dataset["GarageCars"].fillna(dataset["GarageCars"].median())
dataset["GarageYrBlt"] = dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].median())
g = sns.countplot(dataset["Utilities"])
dataset["Utilities"] = dataset["Utilities"].fillna("AllPub")
g = sns.countplot(dataset["MSZoning"])
dataset["MSZoning"] = dataset["MSZoning"].fillna("RL")
g = sns.countplot(dataset["KitchenQual"])
dataset["KitchenQual"] = dataset["KitchenQual"].fillna("TA")
g = sns.countplot(dataset["SaleType"])
dataset["SaleType"] = dataset["SaleType"].fillna("WD")
## 
Function_feat = ["Functional","Exterior2nd","Exterior1st","Electrical"]
multiplot(data = dataset ,features = Function_feat, plottype = "countplot",nrows = 2, ncols = 2, figsize = (11,9), colorize = True)
dataset["Functional"] = dataset["Functional"].fillna("Typ")
dataset["Exterior2nd"] = dataset["Exterior2nd"].fillna("VinylSd")
dataset["Exterior1st"] = dataset["Exterior1st"].fillna("VinylSd")
dataset["Electrical"] = dataset["Electrical"].fillna("SBrkr")

## categorical variables
dataset = dataset.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30',40: 'SubClass_40',
45: 'SubClass_45',50: 'SubClass_50',60: 'SubClass_60',70: 'SubClass_70',
75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',
120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',
190: 'SubClass_190'}})
dataset = dataset.replace({'MoSold': {1: 'Jan', 2: 'Feb',3: 'Mar',
4: 'Apr',5: 'May',6: 'Jun',7: 'Jul',8: 'Aug',9: 'Sep',10: 'Oct',
11: 'Nov',12: 'Dec'}})
dataset['YrSold'] = dataset['YrSold'].astype(str)


## Categorical values
## Ordered
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
dataset["BsmtCond"] = dataset["BsmtCond"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Mn','Av','Gd'],ordered=True)
dataset["BsmtExposure"] = dataset["BsmtExposure"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True)
dataset["BsmtFinType1"] = dataset["BsmtFinType1"].astype(cat_dtype)
dataset["BsmtFinType2"] = dataset["BsmtFinType2"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
dataset["BsmtQual"] = dataset["BsmtQual"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
dataset["ExterCond"] = dataset["ExterCond"].astype(cat_dtype)
dataset["ExterQual"] = dataset["ExterQual"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','MnWw','GdWo','MnPrv','GdPrv'],ordered=True)
dataset["Fence"] = dataset["Fence"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
dataset["FireplaceQu"] = dataset["FireplaceQu"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],ordered=True)
dataset["Functional"] = dataset["Functional"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
dataset["GarageCond"] = dataset["GarageCond"].astype(cat_dtype)
dataset["GarageQual"] = dataset["GarageQual"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Unf','RFn','Fin'],ordered=True)
dataset["GarageFinish"] = dataset["GarageFinish"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
dataset["HeatingQC"] = dataset["HeatingQC"].astype(cat_dtype)
dataset["KitchenQual"] = dataset["KitchenQual"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['N','P','Y'],ordered=True)
dataset["PavedDrive"] = dataset["PavedDrive"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['No','Fa','TA','Gd','Ex'],ordered=True)
dataset["PoolQC"] = dataset["PoolQC"].astype(cat_dtype)
cat_dtype = pd.api.types.CategoricalDtype(categories=['ELO','NoSeWa','NoSewr','AllPub'],ordered=True)
dataset["Utilities"] = dataset["Utilities"].astype(cat_dtype)

## non ordered
dataset = pd.get_dummies(dataset, columns=["Alley","BldgType","CentralAir","Condition1","Condition2","Electrical","Exterior1st","Exterior2nd","Foundation","GarageType","Heating","HouseStyle","LandContour","LandSlope","LotConfig","LotShape","MSZoning","MasVnrType","MiscFeature","Neighborhood","RoofMatl","RoofStyle","SaleCondition","SaleType","Street","MSSubClass",'MoSold','YrSold'], drop_first=True)
## ces variables sont colineaires
dataset = dataset.drop(labels=['MSSubClass_SubClass_150','Condition2_PosN', 'MSSubClass_SubClass_160'],axis = 1)
## print skewed features
skewed_features = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","GarageArea","MasVnrArea", "TotalBsmtSF","1stFlrSF","2ndFlrSF","3SsnPorch","EnclosedPorch", "GrLivArea","LotArea","LowQualFinSF","OpenPorchSF","PoolArea","ScreenPorch","WoodDeckSF"]
multiplot(data = dataset,features = skewed_features,plottype = "distplot", nrows = 4, ncols = 4, figsize = (11,9), colorize = True)

## normalize
skewed_features = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","GarageArea","MasVnrArea", "TotalBsmtSF","1stFlrSF","2ndFlrSF","3SsnPorch","EnclosedPorch", "GrLivArea","LotArea","LowQualFinSF","OpenPorchSF","PoolArea","ScreenPorch","WoodDeckSF"]
for feature in skewed_features:
    dataset[feature] = np.log1p(dataset[feature])
multiplot(data = dataset,features = skewed_features,plottype = "distplot", nrows = 4, ncols = 4, figsize = (11,9), colorize = True)
dataset["SalePrice"] = np.log1p(dataset["SalePrice"])
Y = dataset["SalePrice"]
dataset = dataset.drop(labels="SalePrice",axis = 1)

features = dataset.columns

LotF = dataset["LotFrontage"]
dataset = dataset.drop(labels="LotFrontage",axis= 1)

N = RobustScaler()

N.fit(dataset)
dataset = N.transform(dataset)

## modelization

## Separate train dataset and test dataset
X_train = dataset[:train_len]
test = dataset[train_len:]