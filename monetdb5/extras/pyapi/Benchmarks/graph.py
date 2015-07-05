

import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import sys


arguments = sys.argv
action = arguments[1]
graph_title = arguments[2]
x_axis = arguments[3]
x_unit = 'Unknown'
legend = []
colors = ['blue', 'red', 'green']
color_index = 0

#files
fig, (ax0) = plt.subplots(nrows=1, sharex=True)
for i in range(4, len(arguments)):
    spl = arguments[i].split(':')
    file_title = spl[0]
    file_loc = spl[1]
    f = open(file_loc, 'r')
    line = f.readline()
    x_data = []
    y_data = []
    y_std = []
    current_data = []
    data = False
    while len(line) > 0:
        if ':' in line:
            if len(current_data) > 0:
                list.append(y_data, numpy.mean(current_data))
                list.append(y_std, numpy.std(current_data))
            headers = line.split(',')
            for header in headers:
                if x_axis in header:
                    x_value = -1
                    values = header.split(':')[1].translate(None, '\n\t').split(' ')
                    for value in values:
                        if len(value) > 0:
                            try:
                                x_value = float(value)
                            except:
                                x_unit = value
                    list.append(x_data, x_value)
                    current_data = []
                    data = False
                    break
        elif data:
            try:
                value = float(line.translate(None, ' \t\n'))
                list.append(current_data, value)
            except:
                pass
        elif line.translate(None, ' \t\n').lower() == 'pyapiexecution':
                data = True
        line = f.readline()
    if len(current_data) > 0:
        list.append(y_data, numpy.mean(current_data))
        list.append(y_std, numpy.std(current_data))
    f.close()
    if len(x_data) != len(y_data):
        raise Exception('Wrong input')
    ax0.errorbar(x_data, y_data, yerr=y_std, fmt='-o', label=file_title)
    color_index = color_index + 1
ax0.set_title(graph_title)
ax0.set_xscale('log')
ax0.set_yscale('log')
plt.xlabel(x_axis + ' (' + x_unit + ')')
plt.ylabel('time (s)')
plt.legend()
if action.lower() == 'save':
    plt.savefig(graph_title + ".png", dpi=200)
else:
    plt.show()


