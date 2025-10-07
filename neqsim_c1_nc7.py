#%%
from neqsim import jneqsim
from neqsim.thermo import TPflash
import time
import numpy as np

kij = 0.05

x_y_points = 200


t_min=100
t_max = 300
t_step = (t_max-t_min)/x_y_points

p_max = 500
p_step = p_max / x_y_points

testSystem = jneqsim.thermo.system.SystemPrEos()
# other models can be selected from: https://github.com/equinor/neqsim/tree/master/src/main/java/neqsim/thermo/system
testSystem.addComponent("methane", 70.0)
testSystem.addComponent("n-heptane", 30.0)

testSystem.setMixingRule("classic")

testSystem.getPhase(0).getMixingRule().setBinaryInteractionParameter(0, 1, kij)
testSystem.getPhase(1).getMixingRule().setBinaryInteractionParameter(0, 1, kij)

testSystem.setMultiPhaseCheck(True)

data = np.zeros([x_y_points, x_y_points], dtype=np.uint8)
beta = np.zeros([x_y_points, x_y_points], dtype=np.uint8)

pressures = []


start = time.time()

for i in range(data.shape[0]):
    pres = i * p_step
    testSystem.setPressure(pres, "bara")
    pressures.append(pres)
    temperatures = []

    for j in range(data.shape[1]):
        temp = t_min + j * t_step
        testSystem.setTemperature(temp, "K")
        temperatures.append(testSystem.getTemperature("K"))


        # To calculate mixture density:
        # testSystem.initPhysicalProperties('density')
        # density = testSystem.getDensity('kg/m3')

        try:
            TPflash(testSystem)
            beta[i][j]=testSystem.getBeta()
            data[i][j] = testSystem.getNumberOfPhases()
            if (
                testSystem.hasPhaseType("aqueous")
                and testSystem.hasPhaseType("oil")
                and not testSystem.hasPhaseType("gas")
            ):
                data[i][j] = 4
            if (
                not testSystem.hasPhaseType("aqueous")
                and testSystem.hasPhaseType("oil")
                and not testSystem.hasPhaseType("gas")
            ):
                data[i][j] = 5
            if (
                not testSystem.hasPhaseType("aqueous")
                and not testSystem.hasPhaseType("oil")
                and testSystem.hasPhaseType("gas")
            ):
                data[i][j] = 6
            if (
                testSystem.hasPhaseType("aqueous")
                and testSystem.hasPhaseType("oil")
                and testSystem.hasPhaseType("gas")
            ):
                data[i][j] = 7
            if (
                testSystem.hasPhaseType("aqueous")
                and not testSystem.hasPhaseType("oil")
                and testSystem.hasPhaseType("gas")
            ):
                data[i][j] = 8
            if (
                not testSystem.hasPhaseType("aqueous")
                and testSystem.hasPhaseType("oil")
                and testSystem.hasPhaseType("gas")
            ):
                data[i][j] = 9
            if (
                testSystem.hasPhaseType("oil")
                and testSystem.getNumberOfPhases() == 2
                and not testSystem.hasPhaseType("gas")
                and not testSystem.hasPhaseType("aqueous")
            ):
                data[i][j] = 10
        except:
            data[i][j] = 11

end = time.time()
elapsed = end - start

print("time per flash ", elapsed / (x_y_points * x_y_points) * 1000, " msec")
#%%

start = time.time()
temps2 = np.arange(0,p_max, p_step)
press2 = np.arange(t_min, t_max, t_step)
#%%

import matplotlib.pyplot as plt
import numpy as np

# Assuming the data array and other variables are already populated

# Define a dictionary to map data values to region names
region_labels = {
    4: "Aqueous-Oil",
    5: "Oil only",
    6: "Gas only",
    7: "Gas-Oil-Aqueous",
    8: "Gas-Aqueous",
    9: "Gas-Oil",
    10: "Oil-Oil",
}

# Dictionary to store the coordinates of all points in each region
region_points = {key: [] for key in region_labels.keys()}

# Collect all points for each region
for i in range(x_y_points):
    for j in range(x_y_points):
        region_value = data[i][j]
        if region_value in region_points:
            region_points[region_value].append((temperatures[j], pressures[i]))

# Plot the data
plt.pcolormesh(temperatures, pressures, data, cmap="viridis")
plt.xlabel("Temperature [K]")
plt.ylabel("Pressure [bara]")

# Add one label at the center of each region
for region_value, points in region_points.items():
    if points:
        # Calculate the centroid of the region
        avg_temp = np.mean([p[0] for p in points])
        avg_pres = np.mean([p[1] for p in points])
        # Place the label at the centroid
        plt.text(avg_temp, avg_pres, region_labels[region_value], fontsize=10, ha="center", va="center", color="white")

plt.show()
