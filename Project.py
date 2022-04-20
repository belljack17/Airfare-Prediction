#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:42:21 2021

@author: jackkrolik
"""
#=========================================================================
'''Project Formatting'''
#=========================================================================

import pandas as pd

df1 = pd.read_csv("Q1_ticket.csv")
df2 = pd.read_csv("Q2_ticket.csv")
df3 = pd.read_csv("Q3_ticket.csv")
df4 = pd.read_csv("Q4_ticket.csv")

df_ticket = pd.concat([df1,df2,df3, df4], ignore_index=True)
df_ticket.to_csv("df2019_TICKET")

df = pd.concat([df1,df2,df3, df4], ignore_index=True)
df.columns
df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop('ITIN_ID',axis=1,inplace=True)

df.loc[df["QUARTER"] == 1,"QUARTER"]="Quarter 1"
df.loc[df["QUARTER"] == 2,"QUARTER"]="Quarter 2"
df.loc[df["QUARTER"] == 3,"QUARTER"]="Quarter 3"
df.loc[df["QUARTER"] == 4,"QUARTER"]="Quarter 4"


df = df.drop(df[df.DOLLAR_CRED == 0].index)

df.drop('DOLLAR_CRED',axis=1,inplace=True)
df['TOTAL_FARE'] = df['ITIN_FARE']*df['PASSENGERS']
df.drop('ITIN_FARE',axis=1,inplace=True)
df = df.drop(df[df.TOTAL_FARE == 0].index)

airlines = df["REPORTING_CARRIER"].value_counts()
airlines = airlines.to_frame()
airlines = airlines.reset_index()
airlines = airlines.rename({'index' : 'airline'}, axis = 1)

airports = df['ORIGIN'].value_counts()
airports = airports.to_frame()
airports = airports.reset_index()
airports = airports.rename({'index' : 'airport'}, axis = 1)

df = df[df.groupby(['ORIGIN'])['ORIGIN'].transform('size') > 220000]
df = df[df.groupby(['REPORTING_CARRIER'])['REPORTING_CARRIER'].transform('size') > 100000]

df.to_csv('DATA_NEW.csv', index = False)
df['FPP'] = df['TOTAL_FARE']/df['PASSENGERS']
df = df.drop(df[df.FPP > 1396].index)
df.to_csv("DATA_NEW1.csv")

df = pd.read_csv("DATA_NEW1.csv")
df.columns
mean = df['PASSENGERS'].mean()

std = 3* df['PASSENGERS'].std()

mean + std

import matplotlib.pyplot as plt
plt.hist(df['PASSENGERS'])

df = df.drop(df[df.PASSENGERS > 25].index)

mean = df['DISTANCE'].mean()

std = 3* df['DISTANCE'].std()

mean + std

import matplotlib.pyplot as plt
plt.hist(df['DISTANCE'])

df = df.drop(df[df.DISTANCE > 6616].index)

mean = df['ITIN_YIELD'].mean()

std = 3* df['ITIN_YIELD'].std()

mean + std

import matplotlib.pyplot as plt
plt.hist(df['ITIN_YIELD'])

df = df.drop(df[df.ITIN_YIELD > 1].index)


df.to_csv("DATA_NEW.csv")

#=========================================================================
'''Project Modeling'''
#=========================================================================

import pandas as pd
#import
df = pd.read_csv("DATA_NEW.csv")
df.columns
df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop('BULK_FARE',axis=1,inplace=True) #only unique value is 0, not useful for analysis

df = pd.get_dummies(data=df,drop_first=True)

#define
x=df.drop("FPP",axis=1)
y=df["FPP"]

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)



from sklearn.preprocessing import MinMaxScaler #import
scaler=MinMaxScaler() #initialize
scaler.fit(x_train) #train
scaled_x_train=scaler.transform(x_train)
scaled_x_test=scaler.transform(x_test)


'''LINEAR REGRESSION'''
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() #initialize
lm.fit(scaled_x_train,y_train) #train
y_pred=lm.predict(scaled_x_test)

# from sklearn.model_selection import cross_val_score
# cvs = cross_val_score(lm,scaled_x,y,cv=10)


from sklearn.metrics import r2_score, mean_squared_error
r2_test=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=mse**.5
print("r square is",r2_test) 
print("rmse is", rmse) 

import matplotlib.pyplot as plt
plt.scatter(df["ITIN_YIELD"], lm.predict(x_test), label="predicted")
plt.scatter(scaled_x_test, y_test, label="actuals")
plt.legend()

'''Try TO IMPROVE THE MODEL USING SELECTKBEST'''
#import,initialize,train,transform
from sklearn.feature_selection import SelectKBest,f_regression #import
x_list=[]
y_list=[]
for i in range(1,47):
    bestfeatures=SelectKBest(score_func=f_regression,k=i) #initialize
    bestfeatures.fit(scaled_x_train,y_train) #train

    new_x_train=bestfeatures.transform(scaled_x_train)
    new_x_test=bestfeatures.transform(scaled_x_test)
    
    from sklearn.linear_model import LinearRegression 
    lm=LinearRegression() 
    lm.fit(new_x_train,y_train)
    
    y_pred=lm.predict(new_x_test)
    from sklearn.metrics import r2_score, mean_squared_error
    r2_test=r2_score(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=mse**.5
    y_list.append(rmse)
    x_list.append(i)

print("test r square is",r2_test) 
print("test rmse is", rmse)     

#do a plot between rmse and k value
import seaborn as sns
sns.scatterplot(x=x_list,y=y_list)
import matplotlib.pyplot as plt
plt.plot(x_list,y_list)

#Using k=9
bestfeatures=SelectKBest(score_func=f_regression,k=9) #initialize
bestfeatures.fit(scaled_x_train,y_train) #train

bestfeatures.get_support()

new_x_train=bestfeatures.transform(scaled_x_train)
new_x_test=bestfeatures.transform(scaled_x_test)

#Running Linear Regression on the 9 best features selected
from sklearn.linear_model import LinearRegression 
lm=LinearRegression() 
lm.fit(new_x_train,y_train)

y_pred=lm.predict(new_x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_test=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=mse**.5
#r2 0.679
#rmse 143.183


'''Cross Validation K-Fold'''
from sklearn.model_selection import cross_val_score
cv = cross_val_score(lm,x,y,cv=10,scoring='neg_root_mean_squared_error')
cv2 = cross_val_score(lm,x,y,cv=10)


'''Decision Tree'''
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
rmse_list=[]
r2_list=[]
for i in range(1,12):
    dtr=DecisionTreeRegressor(max_depth=i, random_state=1)
    dtr.fit(x_train,y_train)

    dtr_pred=dtr.predict(x_test)
    rmse=mean_squared_error(y_test,dtr_pred)**0.5
    rmse_list.append(rmse)
    
    r2=r2_score(y_test,dtr_pred)
    r2_list.append(r2)
    
dtr=DecisionTreeRegressor(max_depth=5, random_state=1)
dtr.fit(x_train,y_train)
rmse_5=mean_squared_error(y_test,dtr_pred)
r2_5=r2_score(y_test,dtr_pred)

    
from sklearn.tree import plot_tree
dtr.tree_.max_depth
plot_tree(dtr,max_depth=6)

##tree.plot_tree(dtr, max_depth=10)

from sklearn.metrics import mean_squared_error,r2_score
r2=r2_score(y_test,dtr_pred)
rmse=mean_squared_error(y_test,dtr_pred)**0.5

'''Random Forest'''
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(max_depth=5,random_state=1,n_estimators=100)
rfr.fit(x_train,y_train)


y_pred=rfr.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(y_test,y_pred)
rmse=mse**0.5
r2=r2_score(y_test,y_pred)

sns.scatterplot(y_test,dtr_pred)
plt.xlabel('FPP True Values ')
plt.ylabel('FPP Predictions ')

sns.scatterplot(y_test,y_pred)
plt.xlabel('FPP True Values ')
plt.ylabel('FPP Predictions ')


plot_tree(rfr)

from sklearn.tree import export_graphviz
export_graphviz(rfr)


#=========================================================================
'''Project Visualizations'''
#=========================================================================
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("DATA_NEW.csv")
fig = plt.figure()
df['ORIGIN'].unique()

fig = plt.figure(figsize = (24,16))
sns.relplot(x="ORIGIN", y="FPP",data=df)


plt.figure(figsize=(12,8))###controlsthe length and height)
plt.plot(df["Open"],label="Tesla stock",color="red")
plt.legend(loc="upper right")
plt.xlabel("Date",fontsize=18)
plt.ylabel("Stock price",fontsize=15)
plt.xticks(fontsize=25,rotation=90)
plt.yticks(fontsize=17)

df['QUARTER'].unique()

#graph1
df1 = df[['ORIGIN','FPP']]
df2 = pd.DataFrame(df1['ORIGIN'].value_counts())
df2.index.name = 'origin'
df3 = df1.groupby(df1['ORIGIN']).mean()
df3.index.name = 'origin'
df4 = df2.merge(df3,on='origin')
df4=df4.sort_values(by='ORIGIN',ascending=False)
df4[['ORIGIN']].plot(kind='bar',width=0.20,figsize = (12,8),grid=True,label = 'FPP')
df4['FPP'].plot(secondary_y=True,figsize = (12,8),color='red',grid=True)
df4.set_xlim(df4.index[0], df4.index[-1])
df4.set_title("Relationship between FPP and ORIGIN")

##graph2 
df5 = df[['PASSENGERS','FPP']]
df6 = pd.DataFrame(df5['PASSENGERS'].value_counts())
df6.index.name = 'passenger'
df7 = df5.groupby(df5['PASSENGERS']).mean()
df7.index.name = 'passenger'
df8 = df6.merge(df7,on='passenger')
df8[['PASSENGERS']].plot(kind='bar',figsize = (16,8),grid=True,label = 'PASSENGERS',color='black',xlim=(df8.index[0], df8.index[-1]))
df8['FPP'].plot(secondary_y=True,figsize = (16,8),color='green',grid=True)
plt.ylabel("FPP")

df8.index[0]
xlim=(df8.index[0], df8.index[-1])
#graph3
df9 = df[['QUARTER','FPP']]
df10 = pd.DataFrame(df9['QUARTER'].value_counts())
df10.index.name = 'quarter'
df11 = df9.groupby(df9['QUARTER']).mean()
df11.index.name = 'quarter'
df12= df11.merge(df10,on='quarter')
df12[['QUARTER']].plot(kind='bar',figsize = (12,8),grid=True,color='Orange')
df12['FPP'].plot(secondary_y=True,figsize = (12,8),color='black',grid=True)
plt.ylabel('FPP')

fig, ax = plt.subplots(figsize=[9, 7])

# Plotting the firts line with ax axes

###graph 4

graph2=sns.jointplot(x='PASSENGERS',y="FPP",data=df8)
graph3=sns.regplot(x=df8.index,y="FPP",data=df8)

plt.figure(figsize=(10,6))
sns.countplot(x="ORIGIN", data=df,hue="ONLINE")
df4['FPP'].plot(secondary_y=True,figsize = (12,8),color='red',grid=True)

df13=df.loc[df["ONLINE"]==0][['FPP']]
df14=df.loc[df["ONLINE"]==1][["FPP"]]
plt.figure(figsize=(12,8))
plt.hist(df13["FPP"],label="NOT ONLINE ",bins=20, color="red")
plt.xlabel("NON_ONLINE_PRICE")
plt.ylabel("Frequency")
plt.grid(True)
plt.hist(df14["FPP"],label="ONLINE",bins=20, color="blue")
plt.xlabel("ONLINE_PRICE")
plt.ylabel("Frequency")
plt.grid(True)

##graph5
x_coord=df['DISTANCE']
y_coord=df['FPP']

plt.scatter(x_coord,y_coord)
plt.xlabel("DISTANCE",fontsize=15)
plt.ylabel("FPP",fontsize=15)



plt.text(20, 10,"This is my first text")
plt.xlabel("x lable",fontsize=18)
plt.ylabel("y lable",fontsize=15)



# Creating figure and axis objects using subplots()
fig, ax = plt.subplots(figsize=[9, 7])

# Plotting the firts line with ax axes
ax.plot(df14.index,
        df14['Rate'],label = 'distance',
        color='b', linewidth=2, marker='o')
plt.legend(loc="upper left",fontsize=10)
plt.xticks(rotation=60)
ax.set_xlabel('ORIGIN', fontsize=15)
ax.set_ylabel('DISTANCE',  color='blue', fontsize=15)

# Create a twin axes ax2 using twinx() function
ax2 = ax.twinx()

# Now, plot the second line with ax2 axes
ax2.plot(df14.index,
         df14['FPP'],
         color='orange', linewidth=2, marker='o',label = 'FPP')
plt.legend(loc="upper right",fontsize=10)
ax2.set_ylabel('FPP', color='orange', fontsize=15)

plt.show()

##graph 6
df15=df.loc[df["ONLINE"]==0][['COUPONS']]
df16=df.loc[df["ONLINE"]==1][["COUPONS"]]
plt.figure(figsize=(12,8))
plt.hist(df15["COUPONS"],label="NOT ONLINE ",bins=20, color="red")
plt.xlabel("NON_ONLINE_PRICE")
plt.ylabel("Frequency")
plt.grid(True)
plt.hist(df16["COUPONS"],label="ONLINE",bins=20, color="blue")
plt.xlabel("ONLINE_PRICE")
plt.ylabel("Frequency")
plt.grid(True)