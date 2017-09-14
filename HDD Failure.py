import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import glob, os
from datetime import datetime
from pylab import rcParams
from sklearn.preprocessing import LabelEncoder, Imputer
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

os.chdir("C:\\Users\\Andrew\\Downloads\\hdd\\")

allFiles = glob.glob("*.csv")
df = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0, usecols=[0, 1, 2, 3,4, 20])
    list_.append(df)
df = pd.concat(list_)
df.reset_index(inplace=True)

os.chdir("C:\\Users\\Andrew\\Documents")

df["mindate"] = df["date"].groupby(df["serial_number"]).transform('min')
df["maxdate"] = df["date"].groupby(df["serial_number"]).transform('max')
df["minhours"] = df["smart_9_raw"].groupby(df["serial_number"]).transform('min')
df["maxhours"] = df["smart_9_raw"].groupby(df["serial_number"]).transform('max')
df["nrec"] = df["date"].groupby(df["serial_number"]).transform('count')

df = df[["date", "serial_number","model","capacity_bytes","mindate","maxdate",
        "minhours", "maxhours","nrec","failure"]]

df = df.sort_values("failure",ascending=False)
df = df.drop_duplicates(["serial_number"],keep="first")
df["mindate"] = pd.to_datetime(df["mindate"])
df["maxdate"] = pd.to_datetime(df["maxdate"])
df.reset_index(inplace=True)

#Save off file
df.to_csv("HDD-log.csv", index=False)

df = pd.read_csv("HDD-log.csv")

#check for anything odd with failed
df["failure"].value_counts()

#check for any hard drive that is just too small
df["capacity_bytes"].value_counts()

#drop hdd that are too small of records or too large
df = df.loc[(df['capacity_bytes']<1,000,000,000) & (df["capacity_bytes"]>15,000,000,000)]]

#Create the make/model of the drives
df["make"] = df["model"].apply(lambda x:x.split()[0])
df["model"] = df["model"].apply(lambda x: x.split()[1] if len(x.split())>1
else x)
df.groupby(["make","model"]).size()

#Create the Seagate make and Hitachi
df["make"] = df["make"].apply(lambda x:"SEAGATE" if x[:2]== "ST" else x)
df.loc[df["make"] == "HGST","make"] = "Hitachi"

#adjust to TB size
df["capacity"] = df["capacity_bytes"].apply(lambda x: '{}TB'.format(
    round(
    x/1000000000000,2)))
df.groupby(["capacity"]).size()

#Visual inspection of the data
gp = df.groupby(["make","capacity"]).size().unstack()
sb.heatmap(gp, mask=pd.isnull(gp), robust=True, square=True,cbar=False)

#Shrink to just Hitachi, Seagate, WD
df = df.loc[df["make"].isin(["Hitachi", "Seagate", "WDC"])]

#visual inspection of date start and ending


df['mindateym'] = df['mindate'].apply(lambda x: x.strftime('%Y%m'))
df['maxdateym'] = df['maxdate'].apply(lambda x: x.strftime('%Y%m'))
gp = df.groupby(["mindateym","maxdateym"]).size().unstack()
sb.heatmap(gp, mask=pd.isnull(gp), robust=True, square=True,cbar=False)

ax=df.hist(column='maxhours', by='failed', bins=100, layout=(2,1), log=True, sharex=True, sharey=True)
sb.distplot(df["maxhours"],bins=100,)