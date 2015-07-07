

import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import sys


arguments = sys.argv
action = arguments[1]
graph_title = arguments[2]
file_titles = []
axis = []
graph_data = []
sorted_indices = []
x_axis = "Unknown"
y_axis = "Unknown"

for i in range(3, len(arguments)):
    spl = arguments[i].split(':')
    list.append(file_titles, spl[0])
    file_loc = spl[1]
    f = open(file_loc, 'r')
    title = f.readline().translate(None, '\n').split('\t') # first read the title header
    if len(title) < 7: raise Exception('Invalid header data')
    data = []
    for i in range(5, len(title)):
        list.append(axis, title[i])
        list.append(data, [])
    line = f.readline()
    while len(line) > 0:
        split_data = line.split('\t')
        if len(split_data) < 7: continue #invalid line
        for i in range(5, len(title)):
            list.append(data[i - 5], float(split_data[i]))
        line = f.readline()
    f.close()
    unique = []
    for i in range(0, len(data) - 1):
        list.append(unique, numpy.unique(data[i]))
    sorted_indices = [i[0] for i in sorted(enumerate(unique), key=lambda x:len(x[1]))]
    dictionary = dict()
    for i in range(0, len(sorted_indices)):
        base = [dictionary]
        new_base = []
        while True:
            for j in range(0, len(base)):
                for key in base[j].keys():
                    list.append(new_base, base[j][key])
            if len(new_base) == 0: break
            base = new_base
            new_base = []
        for j in range(0, len(unique[sorted_indices[i]])):
            for dic in base:
                dic[unique[sorted_indices[i]][j]] = dict()
    def recursive_find(dictionary, data, mask, sorted_indices, unique, index = 0):
        if index == len(sorted_indices):
            new_data = numpy.array(data[index])[mask]
            dictionary['mean'] = numpy.mean(new_data)
            dictionary['std'] = numpy.std(new_data)
        else:
            for val in unique[sorted_indices[index]]:
                new_mask = numpy.logical_and(mask, data[sorted_indices[index]]==val)
                recursive_find(dictionary[val], data, new_mask, sorted_indices, unique, index + 1)
    recursive_find(dictionary, numpy.array(data), numpy.array([True] * len(data[0])), sorted_indices, unique)
    list.append(graph_data, dictionary)
    x_axis = axis[sorted_indices[len(sorted_indices)-1]]
    y_axis = axis[len(axis)-1]

def format_subtitle(title, value):
    ind = title.find('(')
    try: value = int(value)
    except: pass
    return title[:ind+1] + str(value) + ' ' + title[ind+1:]

def plot_graphs(graph_data, title, subtitle = "", index = 0):
    if index == len(sorted_indices) - 1:
        if len(subtitle) > 0: subtitle = '\n' + subtitle[:len(subtitle)-1]
        fig, (ax0) = plt.subplots(nrows=1, sharex=True)
        j = 0
        for dictionary in graph_data:
            x_data = sorted(dictionary.keys())
            y_data = []
            y_std = []
            for x_value in x_data:
                list.append(y_data, dictionary[x_value]['mean'])
                list.append(y_std, dictionary[x_value]['std'])
            ax0.errorbar(x_data, y_data, yerr=y_std, fmt='-o', label=file_titles[j])
            j += 1
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.set_title(title + subtitle)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend(bbox_to_anchor=(0,0.83), loc=3)
        if action.lower() == 'save':
            plt.savefig(title + subtitle, format="pdf")
        else:
            plt.show()
    else:
        for key in graph_data[0].keys():
            new_graph_data = []
            for graph in graph_data:
                list.append(new_graph_data, graph[key])
            plot_graphs(new_graph_data, title, subtitle + format_subtitle(axis[sorted_indices[index]], key) + ",",index + 1)

plot_graphs(graph_data, graph_title)


