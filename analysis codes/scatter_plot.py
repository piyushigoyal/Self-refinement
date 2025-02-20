# import matplotlib.pyplot as plt

# data = [
#     {'Cost': [182463.0, 147523.0], 'Accuracy': [32.22, 18.27], 'Label': '0.5,0.5'},
#     {'Cost': [173120.0, 114340.0], 'Accuracy': [54.96, 27.36], 'Label': '0.5,1.5'},
#     {'Cost': [171215.0, 115232.0], 'Accuracy': [64.06, 31.61], 'Label': '0.5,3'},
#     {'Cost': [171465.0, 139225.0], 'Accuracy': [72.55, 31.46], 'Label': '0.5,7'},

#     {'Cost': [182463.0, 167852.0], 'Accuracy': [32.22, 22.06], 'Label': '1.5,0.5'},
#     {'Cost': [173120.0, 138517.0], 'Accuracy': [54.96, 36.92], 'Label': '1.5,1.5'},
#     {'Cost': [171215.0, 136336.0], 'Accuracy': [64.06, 41.93], 'Label': '1.5,3'},
#     {'Cost': [171465.0, 157874.0], 'Accuracy': [72.55, 40.56], 'Label': '1.5,7'},

#     {'Cost': [182463.0, 166099.0], 'Accuracy': [32.22, 22.89], 'Label': '3,0.5'},
#     {'Cost': [173120.0, 133234.0], 'Accuracy': [54.96, 37.68], 'Label': '3,1.5'},
#     {'Cost': [171215.0, 149441.0], 'Accuracy': [64.06, 47.38], 'Label': '3,3'},
#     {'Cost': [171465.0, 166015.0], 'Accuracy': [72.55, 43.13], 'Label': '3,7'},

#     {'Cost': [182463.0, 162595.0], 'Accuracy': [32.22, 23.04], 'Label': '7,0.5'},
#     {'Cost': [173120.0, 134660.0], 'Accuracy': [54.96, 37.68], 'Label': '7,1.5'},
#     {'Cost': [171215.0, 146033.0], 'Accuracy': [64.06, 44.80], 'Label': '7,3'},
#     {'Cost': [171465.0, 172235.0], 'Accuracy': [72.55, 48.21], 'Label': '7,7'},
# ]

# # point_labels = ['ALL', 'FS']

# # plt.figure(figsize=(10, 6))

# # for model in data:
# #     plt.scatter(model['Cost'], model['Accuracy'], label=model['Label'], marker='x')

# #     for i in range(len(point_labels)):
# #         plt.text(model['Cost'][i], model['Accuracy'][i], point_labels[i], fontsize=9, ha='right')

# # # Adding labels and titles
# # plt.xlabel("Cost")
# # plt.ylabel("Accuracy")
# # plt.title("Scatter Plot: Accuracy vs Cost for Multiple Models")
# # plt.legend(loc='best')
# # plt.grid(True)
# # plt.show()

# point_labels = ['ALL', 'FS']

# plt.figure(figsize=(10, 6))

# for model in data:
#     plt.scatter(model['Cost'], model['Accuracy'], label=model['Label'], marker='x')

#     for i in range(len(point_labels)):
#         plt.text(model['Cost'][i], model['Accuracy'][i], point_labels[i], fontsize=9, ha='right')

# # Adding labels and titles
# plt.xlabel("Cost")
# plt.ylabel("Accuracy")
# plt.title("Scatter Plot: Accuracy vs Cost for Multiple Models")

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
    # {'Cost': [1719], 'Accuracy': [35.811], 'Label': '2', 'FS_Cost': [3414, 4204], 'FS_Accuracy': [18.59, 18.74]},
    # {'Cost': [3938], 'Accuracy': [66.91], 'Label': '7', 'FS_Cost': [4937, 5816], 'FS_Accuracy': [35.89, 36.4]},
    # {'Cost': [774], 'Accuracy': [42.88], 'Label': '0.5', 'FS_Cost': [1011, 1393, 1717, 2578], 'FS_Accuracy': [20.56, 24.84, 25.30, 24.63]},
    # {'Cost': [1811], 'Accuracy': [66.72], 'Label': '1.5', 'FS_Cost': [1425, 1854, 2137, 3059], 'FS_Accuracy': [33.74, 37.11, 39.02, 42.80]},
    # {'Cost': [2387], 'Accuracy': [74.64], 'Label': '3', 'FS_Cost': [2030, 2478, 2924, 3715], 'FS_Accuracy': [38.08, 46.41, 49, 47.57]},
    # {'Cost': [3385], 'Accuracy': [82.71], 'Label': '7', 'FS_Cost': [5440, 5935, 6387, 7355], 'FS_Accuracy': [31.68, 34.98, 39.17, 40.72]}

    #single generation
    # {'Cost': [1286], 'Accuracy': [26.99], 'Label': '2', 'FS_Cost': [1088, 1644], 'FS_Accuracy': [18.27, 22.06]},
    # {'Cost': [3784], 'Accuracy': [54.96], 'Label': '7', 'FS_Cost': [2518, 3358], 'FS_Accuracy': [28.88, 36.46]},
    {'Cost': [525], 'Accuracy': [32.22], 'Label': '0.5', 'FS_Cost': [479, 607, 757, 1119], 'FS_Accuracy': [18.27, 22.06, 22.89, 23.04]},
    # {'Cost': [853], 'Accuracy': [54.96], 'Label': '1.5', 'FS_Cost': [687, 891, 1014, 1392], 'FS_Accuracy': [27.36, 36.92, 37.68, 37.68]},
    # {'Cost': [1444], 'Accuracy': [64.06], 'Label': '3', 'FS_Cost': [1098, 1360, 1620, 1963], 'FS_Accuracy': [31.61, 41.93, 47.38, 44.80]},
    # {'Cost': [2833], 'Accuracy': [72.55], 'Label': '7', 'FS_Cost': [2449, 2845, 3130, 3605], 'FS_Accuracy': [31.46, 40.56, 43.13, 48.21]}
]

# Define different markers for ALL points and colors for FS points
all_markers = ['o', 's', 'D', '^']  # Circle, Square, Diamond, Triangle
fs_colors = ['blue', 'green', 'orange', 'red']  # Different colors for FS
models = ['0.5', '1.5','3','7']
# models = ['2', '7']

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
plt.xlim(400, 1200)  # X-axis limits from 100K to 200K
plt.ylim(15, 40)
# plt.xticks(fontsize=16)  # X-axis tick labels font size
# plt.yticks(fontsize=16) # Y-axis tick labels font size
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Put legend outside of plot
# Skip some labels on both axes to reduce clutter
plt.xticks(np.arange(400, 1200, 500), fontsize=30)  # Adjust step size as needed
plt.yticks(np.arange(15, 40, 10), fontsize=30)
plt.tick_params(axis='x', which='major', pad=15)  # Adjust X-axis padding
plt.tick_params(axis='y', which='major', pad=15)  # Adjust Y-axis padding
plt.grid(True)
# plt.legend(loc='upper right',fontsize=16)  # Put legend inside the plot
plt.legend(loc='upper right', fontsize=20, markerscale=0.8, labelspacing=0.4, handletextpad=0.5)
plt.tight_layout()
plt.show()