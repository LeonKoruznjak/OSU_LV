import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    

data = pd.read_csv('data_C02_emission.csv')

#a)
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=50)
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('No. of cars')
plt.title('CO2 Emissions histogram')
plt.show()

#b)
cityConsumtion = data['Fuel Consumption City (L/100km)']
emission = data['CO2 Emissions (g/km)']
plt.scatter(cityConsumtion, emission, color = 'tab:pink', marker = '.')
plt.show()

#c)
grouped_fuel_type = data.groupby('Fuel Type')

grouped_fuel_type.boxplot(column=['Fuel Consumption Hwy (L/100km)'])
plt.show()

#d)
grouped_fuel_type['Make'].count().plot(kind="bar")
plt.ylabel("No. of cars with fuel type")
plt.show()

#e)
grouped_cylinders = data.groupby("Cylinders").mean()
grouped_cylinders["CO2 Emissions (g/km)"].plot(kind="bar")
plt.ylabel("Average CO2 Emissions (g/km)")
plt.show()
