import pandas as pd
import numpy as np

data=pd.read_csv('data_C02_emission.csv')

# a)
print("Broj mjerenja: ", len(data))
print(data.info())
data.drop_duplicates()
data = data . reset_index ( drop = True )
for col in ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']:
    data[col] = data[col].astype('category')



#b)

cityConsumptionSort = data.sort_values(by='Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']]
print("3 lowes city consumption cars:\n", cityConsumptionSort.head(3))
print("\n3 highest city consumption cars:\n", cityConsumptionSort.tail(3))


#c)

engine_size_interval = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print("Broj ovih vozila je", engine_size_interval['Engine Size (L)'].count())
print("Prosječna veličina motora je", round(engine_size_interval['Engine Size (L)'].mean(), 2), "L")

#d)

audies = data[data['Make'] == 'Audi']
print("Broj mjerenja za marku Audi: ", len(audies))
audies4Cylinder = audies[audies['Cylinders'] == 4]
print("4 cylinders audies emmision: ",audies4Cylinder['CO2 Emissions (g/km)'].mean())


#E
groupedByCylinders = data.groupby(data.Cylinders)
print("Average emmisions: ", groupedByCylinders['CO2 Emissions (g/km)'].mean())

#f)
dieselVehicles = data[data['Fuel Type'] == 'D']
gasolineVehicles = data[data['Fuel Type'] == 'X']
print("Diesel mean city consumption: ", dieselVehicles['Fuel Consumption City (L/100km)'].mean())
print("Gasoline mean city consumption: ",gasolineVehicles['Fuel Consumption City (L/100km)'].mean())
print("Diesel median city consumption: ", dieselVehicles['Fuel Consumption City (L/100km)'].median())
print("Gasoline median city consumption: ",gasolineVehicles['Fuel Consumption City (L/100km)'].median())

#g)
FourCylinderDiesels = dieselVehicles[dieselVehicles['Cylinders'] == 4 ]
print("4 cylinder diesel with largest city consumption:\n", FourCylinderDiesels.sort_values(by='Fuel Consumption City (L/100km)').tail(1))

#h)
print("No. of vehicles with manual transmition: ", len(data[data['Transmission'].str.startswith('M')]))

#i)
print(data.corr(numeric_only=True))
