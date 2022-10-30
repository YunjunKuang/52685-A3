import xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

train=pd.read_csv("./data/train.csv")
print("columns:", train.columns)

print(train.head())

print(train.describe())

null_columns = []
for c in train.columns:
    if train[c].isnull().any():
        print(c, train[c].isnull().sum())
        null_columns.append(c)

print("number of columns containing null: ", (len(null_columns)))

sns.heatmap(train.isnull(), yticklabels=False, \
				cbar=False, cmap='coolwarm' )
plt.show()
plt.clf()

train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
train.drop(['Alley'],axis=1,inplace=True)
train['MasVnrType']=train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
train['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['BsmtExposure']=train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType1']=train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])
train['BsmtFinType2']=train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train['Electrical']=train['Electrical'].fillna(train['Electrical'].mode()[0])
train['FireplaceQu']=train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])
train['GarageType']=train['GarageType'].fillna(train['GarageType'].mode()[0])
train['GarageYrBlt']=train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0])
train['GarageFinish']=train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageQual']=train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageCond']=train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
train.drop(['Id'],axis=1,inplace=True)

str_columns = []
for c in train.columns:
    if pd.api.types.is_string_dtype(train.loc[:,c]):
#         print(c)
        str_columns.append(c)

def category_onehot_multcols(final_df, multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        # print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

final_df=category_onehot_multcols(train, str_columns)

y=np.array(final_df["SalePrice"])
X=np.array(final_df.drop(["SalePrice"],axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

cl=xgboost.XGBRegressor()
cl.fit(X_train_scaler,y_train)

y_pred=cl.predict(X_test_scaler)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Only feature scale: R2 score:", r2)
print("Only feature scale: MSE:", mse)

plt.plot(y_pred, label="pred")
plt.plot(np.array(y_test), label="label")
plt.title("Only feature scale")
plt.legend()
plt.show()
plt.clf()


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train_scaler)
X_test_poly = poly.fit_transform(X_test_scaler)

cl=xgboost.XGBRegressor()
cl.fit(X_train_poly,y_train)

y_pred=cl.predict(X_test_poly)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Feature scale and expension: R2 score:", r2)
print("Feature scale and expension: MSE:", mse)

plt.plot(y_pred, label="pred")

plt.plot(np.array(y_test), label="label")
plt.title("Feature scale and expension")
plt.legend()
plt.show()
plt.clf()
