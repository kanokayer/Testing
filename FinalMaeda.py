# coding: utf-8
get_ipython().magic('cd C:/Users/Robin Junior/Documents/My Data Sources/Final')
import pandas as pd
import numpy as np
get_ipython().system('pip install virtualenv')
databay = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//BayCounty.csv", usecols=[0,6])
##Overwriting to include 'Common_Name' instead of scientific name
databay = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//BayCounty.csv", usecols=[0,5])
databrevard = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//BrevardCounty.csv", usecols=[0,5])

databroward = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//BrowardCounty.csv", usecols=[0,5])

datacollier = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//CollierCounty.csv", usecols=[0,5])

dataduval = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//DuvalCounty.csv", usecols=[0,5])

dataescambia = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//EscambiaCounty.csv", usecols=[0,5])
datahillsborough = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//HillsboroughCounty.csv", usecols=[0,5])

datalee = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//LeeCounty.csv", usecols=[0,5])

datamiamidade = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//MiamiDadeCounty.csv", usecols=[0,5])

dataorange = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//OrangeCounty.csv", usecols=[0,5])

datapalmbeach = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//PalmBeachCounty.csv", usecols=[0,5])

datapinellas = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//PinellasCounty.csv", usecols=[0,5])

datasarasota = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//SarasotaCounty.csv", usecols=[0,5])

dataseminole = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//SeminoleCounty.csv", usecols=[0,5])

datastjohns = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//StJohnsCounty.csv", usecols=[0,5])

datastlucie = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//StLucieCounty.csv", usecols=[0,5])

datavolusia = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//VolusiaCounty.csv", usecols=[0,5])
##17 FL county's imported with #'s of non-native plants
datamerged = pd.concat([datapalmbeach, datahillsborough, dataseminole, datapinellas, datastjohns, datasarasota, dataescambia, databay, dataorange, datamiamidade, databroward, datavolusia, databrevard, datastlucie, datacollier, datalee, dataduval], axis=0)
df = datamerged
print(df.head)
get_ipython().magic('save -f FinalMaeda 1-999999')
df.groupby('County', as_index=False)['Common_Name'].count()
df2 = df.groupby('County', as_index=False)['Common_Name'].count()
print(df2)
get_ipython().magic('save -f FinalMaeda 1-999999')
df2=df2.rename(columns = {'Common_Name':'Total_Invasives'})
print(df2)
get_ipython().magic('save -f FinalMaeda 1-999999')
dataflights = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//IntFlightData.csv", usecols=[0,1,3])
print(dataflights.head)
dataflights2 = dataflights['DEST_CITY_NAME'].apply(lambda x: pd.Series(x.split(',')))
print(dataflights2)
dataflights3.'Tot_Passegers'.groupby('Dest_City_Name', as_index=False)['Passengers'].sum()
dataflights3.['Tot_Passegers'].groupby('Dest_City_Name', as_index=False)['Passengers'].sum()
dataflights3.Tot_Passengers = dataflights3.groupby('Dest_City_Name', as_index=False)['PASSENGERS'].sum()
dataflights2.Tot_Passengers = dataflights3.groupby('Dest_City_Name', as_index=False)['PASSENGERS'].sum()
dataflights3 = dataflights3.groupby('DEST_CITY_NAME', as_index=False)['PASSENGERS', 'FREIGHT'].count()
dataflights3 = dataflights2.groupby('DEST_CITY_NAME', as_index=False)['PASSENGERS', 'FREIGHT'].count()
print(dataflights2.head)
dataflights5 = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//IntFlightData.csv", usecols=[0,1,3])
dataflights6 = pd.dataflights5(dataflights5.DEST_CITY_NAME.str.split(' ',1).tolist(), columns = ['County','State'])
dataflights6 = pd.DataFrame(dataflights5.DEST_CITY_NAME.str.split(' ',1).tolist(), columns = ['County','State'])
print(dataflights6)
dataflights6 = pd.DataFrame(dataflights5.DEST_CITY_NAME.str.split(',',1).tolist(), columns = ['County','State'])
print(dataflights6)
get_ipython().magic('save -f FinalMaeda 1-999999')
dataflights7 = dataflights6.replace({'Miami': 'MiamiDade'}, {'Orlando': 'Orange'}, regex=True)
print(dataflights7)
dataflights7 = dataflights6.replace({'Miami': 'MiamiDade', 'Orlando': 'Orange'}, regex=True)
print(dataflights7)
dataflights8 = dataflights7.replace({'West Palm Beach/Palm Beach': 'PalmBeach', 'Tampa': 'Hillsborough', 'Sanford': 'Seminole', 'St. Petersburg': 'Pinellas', 'St. Augustine': 'StJohns', 'Sarasota/Bradenton': 'Sarasota', 'Pensacola': 'Escambia', 'Panama City': 'Bay', 'Fort Lauderdale': 'Broward', 'Daytona Beach': 'Volusia', 'Cocoa Beach': 'Brevard', 'Fort Pierce': 'StLucie', 'Naples': 'Collier','Fort Myers': 'Lee', 'Jacksonville': 'Duval', 'Miami': 'MiamiDade', 'Orlando': 'Orange'}, regex=True)
print(dataflights8.head)
dataflights7 = dataflights6.replace({'West Palm Beach/Palm Beach': 'PalmBeach', 'Tampa': 'Hillsborough', 'Sanford': 'Seminole', 'St. Petersburg': 'Pinellas', 'St. Augustine': 'StJohns', 'Sarasota/Bradenton': 'Sarasota', 'Pensacola': 'Escambia', 'Panama City': 'Bay', 'Fort Lauderdale': 'Broward', 'Daytona Beach': 'Volusia', 'Cocoa Beach': 'Brevard', 'Fort Pierce': 'StLucie', 'Naples': 'Collier','Fort Myers': 'Lee', 'Jacksonville': 'Duval', 'Miami': 'MiamiDade', 'Orlando': 'Orange'}, regex=True)
print(dataflights7)
get_ipython().magic('save -f FinalMaeda 1-999999')
value_list = ['FL']
dataflights8 = dataflights7[dataflights7.State.isin(value_list)]
print(dataflights8)
print(dataflights7)
dataflights7['FL] == True
dataflights7['FL'] == True
dataflights7[dataflights7['FL'] == True]
dataflights8 = dataflights7.loc[dataflights7['State'] == ['FL']]
import numpy as np
dataflights8 = select_rows(dataflights7,['FL'])
dataflights7.PASSENGERSTOT = dataflights7.groupby('County', as_index=False)['PASSENGERS'].sum()
datatrade = pd.read_csv("C://Users//Robin Junior//Documents//My Data Sources//IntFlightData.csv")
print(datatrade.head)
datatrade1 = pd.DataFrame(datatrade.DEST_CITY_NAME.str.split(',',1).tolist(), columns = ['County','State'])
print(datatrade1.head)
pd.DataFrame(datatrade.DEST_CITY_NAME.str.split(',',1).tolist(), columns = ['County','State'])
print(datatrade)
print(datatrade.head)
datatrade2 = pd.DataFrame(datatrade.DEST_CITY_NAME.str.split(',',1).tolist(), columns = ['County','State'])
print(datatrade2.head)
datatrade['County'] = datatrade['DEST_CITY_NAME'].str.rpartition(',')[0].str.replace(",", " ")
datatrade['State']    = datatrade['DEST_CITY_NAME'].str.rpartition(',')[2]
df[['PASSENGERS','FREIGHT', 'MAIL', 'County', 'State', 'Year']]
datatrade['County'] = datatrade['DEST_CITY_NAME'].str.rpartition(',')[0].str.replace(",", " ")
datatrade['State']    = datatrade['DEST_CITY_NAME'].str.rpartition(',')[2]
df[['PASSENGERS','FREIGHT', 'MAIL', 'County', 'State', 'Year']]
print(datatrade)
get_ipython().magic('save -f FinalMaeda 1-999999')
datatrade.PASSENGERTOTAL = datatrade.groupby('County', as_index=False)['PASSENGERS'].sum()
print(datatrade)
datatrade2.PASSENGERTOTAL = datatrade.groupby('County', as_index=False)['PASSENGERS'].sum()
print(datatrade2)
datatrade.PASSENGERTOT.groupby('County', as_index=False)['PASSENGERS'].sum()
datatrade1 = datatrade.groupby('County', as_index=False)['PASSENGERS'].sum()
print(datatrade1)
print(datatrade)
datatrade1 = datatrade[datatrade['State'].str.contains("FL")]
print(datatrade1)
get_ipython().magic('save -f FinalMaeda 1-999999')
datatrade2 = datatrade1.replace({'West Palm Beach/Palm Beach': 'PalmBeach', 'Tampa': 'Hillsborough', 'Sanford': 'Seminole', 'St. Petersburg': 'Pinellas', 'St. Augustine': 'StJohns', 'Sarasota/Bradenton': 'Sarasota', 'Pensacola': 'Escambia', 'Panama City': 'Bay', 'Fort Lauderdale': 'Broward', 'Daytona Beach': 'Volusia', 'Cocoa Beach': 'Brevard', 'Fort Pierce': 'StLucie', 'Naples': 'Collier','Fort Myers': 'Lee', 'Jacksonville': 'Duval', 'Miami': 'MiamiDade', 'Orlando': 'Orange'}, regex=True)
print(datatrade2)
get_ipython().magic('save -f FinalMaeda 1-999999')
datatrade3.TOTPASSENGER = datatrade2.groupby('DEST_CITY_NAME', as_index=False)['PASSENGERS'].sum()
datatrade2.TOTPASSENGER = datatrade2.groupby('DEST_CITY_NAME', as_index=False)['PASSENGERS'].sum()
print(datatrade2)
datatrade3 = datatrade2.groupby('DEST_CITY_NAME', as_index=False)['PASSENGERS'].sum()
print(datatrade3)
datatrade3 = datatrade2.groupby('County', as_index=False)['PASSENGERS'].sum()
print(datatrade3)
datatrade4 = datatrade2.groupby('County', as_index=False)['FREIGHT'].sum()
print(datatrade4)
get_ipython().magic('save -f FinalMaeda 1-999999')
print(df2)
tourtrade = pd.merge(pd.merge(df2,datatrade3,on='County'),datatrade4,on='County')
print(tourtrade)
get_ipython().magic('save -f FinalMaeda 1-999999')
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import scipy.stats
import seaborn
fig = plt.figure(figsize=(17,5))

fig.add_subplot(121)
seaborn.regplot(x="Total_Invasives", y="PASSENGERS", fit_reg=True, data=tourtrade);
plt.xlabel('Total Invasives 2017');
plt.ylabel('Total Int. Passengers');
plt.title('Tourism vs. Trade');
fig.tight_layout()
plt.show()
fig = plt.figure(figsize=(17,5))

fig.add_subplot(121)
seaborn.regplot(x="Total_Invasives", y="PASSENGERS", fit_reg=True, data=tourtrade);
plt.xlabel('Total Invasives 2017');
plt.ylabel('Total Int. Passengers');
plt.title('Tourism and Plant Invasion');

fig.add_subplot(121)
seaborn.regplot(x="Total_Invasives", y="FREIGHT", fit_reg=True, data=tourtrade);
plt.xlabel('Total Invasives 2017');
plt.ylabel('Total Int. Freight');
plt.title('Trade and Plant Invasion');
fig.tight_layout()
plt.show()
fig = plt.figure(figsize=(17,5))

fig.add_subplot(121)
seaborn.regplot(x="Total_Invasives", y="PASSENGERS", fit_reg=True, data=tourtrade);
plt.xlabel('Total Invasives 2017');
plt.ylabel('Total Int. Passengers');
plt.title('Tourism and Plant Invasion');

fig.add_subplot(122)
seaborn.regplot(x="Total_Invasives", y="FREIGHT", fit_reg=True, data=tourtrade);
plt.xlabel('Total Invasives 2017');
plt.ylabel('Total Int. Freight');
plt.title('Trade and Plant Invasion');
fig.tight_layout()
plt.show()
get_ipython().magic('save -f FinalMaeda 1-999999')
print ('Tourism and Plant Invasion')
print (scipy.stats.pearsonr(tourtrade['PASSENGERS'], tourtrade['Total_Invasives']))

print ('Trade and Plant Invasion')
print (scipy.stats.pearsonr(tourtrade['FREIGHT'], tourtrade['Total_Invasives']))
get_ipython().magic('save -f FinalMaeda 1-999999')
###Pearson r results: Tourism and Plant Invasion (0.698153257112203, 0.0018282275958910359), Trade and Plant Invasion (0.68245302035562039, 0.0025402856333718027). These results show that there is 
###a positive correlation between both the number of passengers and the amount of freight arriving internationally and the amount of non-native plant invasion at the county level in Florida. The p-values for both tourism and trade correlations are less than 1%, indicating that there is a less than 1% chance that this result is random. 
###Calculating the correlation coefficient can determine variability:
###Correlation coefficient for Tourism
0.7**2
###Correlation coefficient for Trade
###More precise correlation coefficient for Tourism
0.698**2
###More precise correlation coefficient for Trade
0.682**2
###Trade can help determine about 48.7% of the variability in plant invasion in Florida counties, while trade can explain approximately 46.5%.
get_ipython().magic('save -f FinalMaeda 1-999999')
