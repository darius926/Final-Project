import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg
import statsmodels.formula.api as sm_api
import numpy as np
import seaborn as sns

filename = "C:/Users/dariu/OneDrive/Desktop/wind_turbines.csv"

df=pd.read_csv(filename)
print(df.describe())
df.describe()
print(df.columns) 

y=df['Turbine.Capacity']

x=df[['Turbine.Hub_Height', 'Turbine.Rotor_Diameter', 'Turbine.Swept_Area', 'Turbine.Total_Height']]
df.describe()
print (x.describe())

x=sm.add_constant(x)

print(x)
model=sm.OLS(y,x).fit()

results = model=sm.OLS(y,x).fit()
print(model.summary())


plt.show()
df.describe()
df.plot.scatter(x="Turbine.Hub_Height",y="Turbine.Capacity")
plt.show()
df.plot.scatter(x="Turbine.Rotor_Diameter",y="Turbine.Capacity")
plt.show()
df.plot.scatter(x="Turbine.Swept_Area",y="Turbine.Capacity")
plt.show()
df.plot.scatter(x="Turbine.Total_Height",y="Turbine.Capacity")
plt.show()
df.plot.scatter(x="Turbine.Total_Height",y="Site.State")
plt.show()
sns.boxplot(x="Site.State", y="Turbine.Capacity", data=df)
plt.show()
sns.heatmap(x="Site.State", y="Turbine.Capacity", data=df)
plt.show()

df.describe()
df.plot.scatter(x="independent",y="dependent") 
plt.scatter(x,y)
plt.show()
print(type(df))

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 0, ax=ax)
plt.legend()
plt.show()
summary_stats = df.describe()
print(summary_stats)

plt.scatter(x,y)
plt.show()
