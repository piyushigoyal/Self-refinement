# import matplotlib.pyplot as plt

# data = [
#     {'Cost': [182463.0, 236129.0], 'Accuracy': [32.22, 29.64], 'Label': '0.5,0.5'},
#     {'Cost': [173120.0, 198147.0], 'Accuracy': [54.96, 43.51], 'Label': '0.5,1.5'},
#     {'Cost': [171215.0, 200190.0], 'Accuracy': [64.06, 50.34], 'Label': '0.5,3'},
#     {'Cost': [171465.0, 224042.0], 'Accuracy': [72.55, 53.44], 'Label': '0.5,7'},

#     {'Cost': [182463.0, 240487.0], 'Accuracy': [32.22, 33.58], 'Label': '1.5,0.5'},
#     {'Cost': [173120.0, 195232.0], 'Accuracy': [54.96, 51.55], 'Label': '1.5,1.5'},
#     {'Cost': [171215.0, 197374.0], 'Accuracy': [64.06, 59.36], 'Label': '1.5,3'},
#     {'Cost': [171465.0, 219908.0], 'Accuracy': [72.55, 62.39], 'Label': '1.5,7'},

#     {'Cost': [182463.0, 240361.0], 'Accuracy': [32.22, 35.17], 'Label': '3,0.5'},
#     {'Cost': [173120.0, 190853.0], 'Accuracy': [54.96, 53.98], 'Label': '3,1.5'},
#     {'Cost': [171215.0, 197801.0], 'Accuracy': [64.06, 60.65], 'Label': '3,3'},
#     {'Cost': [171465.0, 222334.0], 'Accuracy': [72.55, 62.47], 'Label': '3,7'},

#     {'Cost': [182463.0, 241135.0], 'Accuracy': [32.22, 36.08], 'Label': '7,0.5'},
#     {'Cost': [173120.0, 193868.0], 'Accuracy': [54.96, 56.10], 'Label': '7,1.5'},
#     {'Cost': [171215.0, 199543.0], 'Accuracy': [64.06, 64.44], 'Label': '7,3'},
#     {'Cost': [171465.0, 217131.0], 'Accuracy': [72.55, 66.33], 'Label': '7,7'},
# ]

# point_labels = ['ALL', 'FS']

# plt.figure(figsize=(10, 6))

# for model in data:
#     plt.scatter(model['Cost'], model['Accuracy'], label=model['Label'], marker='o')

#     for i in range(len(point_labels)):
#         plt.text(model['Cost'][i], model['Accuracy'][i], point_labels[i], fontsize=9, ha='right')

# # Adding labels and titles
# plt.xlabel("Cost")
# plt.ylabel("Accuracy")
# plt.title("Scatter Plot (replaced): Accuracy vs Cost for Multiple Models")
# # plt.legend(loc='best')
# # plt.grid(True)
# # plt.show()


# # Adjusting axis limits to improve visibility
# plt.xlim(min([min(d['Cost']) for d in data]) * 0.95, max([max(d['Cost']) for d in data]) * 1.05)
# plt.ylim(min([min(d['Accuracy']) for d in data]) * 0.95, max([max(d['Accuracy']) for d in data]) * 1.05)

# plt.grid(True)
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Put legend outside of plot
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Updated data for ALL and FS points, including the remaining points
data = [
    #maj voting
    # {'Cost': [1719], 'Accuracy': [35.811], 'Label': '2', 'FS_Cost': [3807, 4645], 'FS_Accuracy': [26.7, 26.19]},
    # {'Cost': [3938], 'Accuracy': [66.91], 'Label': '7', 'FS_Cost': [6027, 6870], 'FS_Accuracy': [43.39, 48.59]},
    # {'Cost': [774], 'Accuracy': [42.88], 'Label': '0.5', 'FS_Cost': [1224, 1567, 1885, 2741], 'FS_Accuracy': [27.8, 32.62, 31.94, 33.64]},
    # {'Cost': [1811], 'Accuracy': [66.72],'Label': '1.5', 'FS_Cost': [1822, 2144, 2465, 3307], 'FS_Accuracy': [44.1, 48.55, 53.43, 52.42]},
    # {'Cost': [2387], 'Accuracy': [74.64], 'Label': '3', 'FS_Cost': [2720, 3006, 3320, 4179], 'FS_Accuracy': [49.2, 58, 60.29, 63.59]},
    # {'Cost': [3385], 'Accuracy': [82.71], 'Label': '7', 'FS_Cost': [6614, 6840, 7158, 8016], 'FS_Accuracy': [45.16, 50.99, 52.21, 54.04]}

    #single gen
    {'Cost': [1286], 'Accuracy': [26.99], 'Label': '2', 'FS_Cost': [1547, 2145], 'FS_Accuracy': [24.18, 28.05]},
    # {'Cost': [3784], 'Accuracy': [54.96], 'Label': '7', 'FS_Cost': [3808, 4408], 'FS_Accuracy': [38.66, 48.36]},
    # {'Cost': [525], 'Accuracy': [32.22], 'Label': '0.5', 'FS_Cost': [693, 788, 937, 1310], 'FS_Accuracy': [29.64, 33.58, 35.17, 36.08]},
    # {'Cost': [853], 'Accuracy': [54.96], 'Label': '1.5', 'FS_Cost': [1102, 1172, 1314, 1685], 'FS_Accuracy': [43.51, 51.55, 53.98, 56.10]},
    # {'Cost': [1444], 'Accuracy': [64.06], 'Label': '3', 'FS_Cost': [1818, 1878, 2030, 2416], 'FS_Accuracy': [50.34, 59.36, 60.65, 64.44]},
    # {'Cost': [2833], 'Accuracy': [72.55], 'Label': '7', 'FS_Cost': [3867, 3882, 4071, 4355], 'FS_Accuracy': [53.44, 62.39, 62.47, 66.33]}
]

# Define different markers for ALL points and colors for FS points
all_markers = ['o', 's', 'D', '^']  # Circle, Square, Diamond, Triangle
fs_colors = ['blue', 'green', 'orange', 'red']  # Different colors for FS
# models = ['0.5', '1.5','3','7']
models = ['2', '7']

plt.figure(figsize=(7, 6)) 

# Plot ALL and FS points
for idx, model in enumerate(data):
    # Plot ALL points with distinct markers
    plt.scatter(model['Cost'], model['Accuracy'], label=model['Label'] + "B - ALL", marker=all_markers[idx], color='black', s=200)

    # Plot FS points with distinct colors and larger markers
    for j in range(len(model['FS_Cost'])):  # Loop through the 4 FS points
        plt.scatter(model['FS_Cost'][j], model['FS_Accuracy'][j], label=f"Asker {models[j]}B", marker='o', color=fs_colors[j], s=250)  # Increase size for visibility


# Adding labels and titles
plt.xlabel("Time (s)", fontsize=30, labelpad=10)
plt.ylabel("Accuracy", fontsize=30, labelpad=10)
# plt.title("Scatter Plot: Accuracy vs Time on All Samples and only correct FS", fontsize=20)
plt.xlim(1000, 2500)  # X-axis limits from 100K to 200K
plt.ylim(20, 40)
# plt.xticks(fontsize=16)  # X-axis tick labels font size
# plt.yticks(fontsize=16) # Y-axis tick labels font size
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Put legend outside of plot
# Skip some labels on both axes to reduce clutter
plt.xticks(np.arange(1000, 2500, 500), fontsize=30)  # Adjust step size as needed
plt.yticks(np.arange(20, 40, 10), fontsize=30)
plt.tick_params(axis='x', which='major', pad=15)  # Adjust X-axis padding
plt.tick_params(axis='y', which='major', pad=15)  # Adjust Y-axis padding
plt.grid(True)
# plt.legend(loc='upper right',fontsize=16)  # Put legend inside the plot
plt.legend(loc='upper right', fontsize=20, markerscale=0.8, labelspacing=0.4, handletextpad=0.5)
plt.tight_layout()
plt.show()