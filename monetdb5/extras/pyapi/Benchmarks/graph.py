

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
y_axis = []
x_log = False
y_log = False
line_plot = False
fill_time = False
data_start = 3
graph_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
for i in range(3, len(arguments)):
    if '-xlog' in arguments[i]: x_log = True
    elif '-ylog' in arguments[i]: y_log = True
    elif '-lineplot' in arguments[i]: line_plot = True
    else:
        data_start = i
        break

measurements = 0

for i in range(data_start, len(arguments)):
    axis = []
    measurements = 0
    spl = arguments[i].split(':')
    list.append(file_titles, spl[0])
    file_loc = spl[1]
    f = open(file_loc, 'r')
    title = f.readline().translate(None, '\n').split('\t') # first read the title header
    if len(title) < 7: raise Exception('Invalid header data')
    data = []
    for i in range(5, len(title)):
        if '[MEASUREMENT]:' in title[i]: measurements = measurements + 1
        list.append(axis, title[i].split(':')[1])
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
    for i in range(0, len(data) - measurements):
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
            for m in range(0, measurements):
                new_data = numpy.array(data[index + m])[mask]
                dictionary['mean' + str(m)] = numpy.mean(new_data)
                dictionary['std' + str(m)] = numpy.std(new_data)
        else:
            for val in unique[sorted_indices[index]]:
                new_mask = numpy.logical_and(mask, data[sorted_indices[index]]==val)
                recursive_find(dictionary[val], data, new_mask, sorted_indices, unique, index + 1)
    recursive_find(dictionary, numpy.array(data), numpy.array([True] * len(data[0])), sorted_indices, unique)
    list.append(graph_data, dictionary)
    x_axis = axis[sorted_indices[len(sorted_indices)-1]]
    y_axis = axis[len(axis)-measurements:]

def format_subtitle(title, value):
    ind = title.find('(')
    try: value = int(value)
    except: pass
    return title[:ind+1] + str(value) + ' ' + title[ind+1:]

bottom_value = None if not y_log else 1e-6
def width_log(x, graphnr, graphs):
    import math
    factor = math.log(x, 10) + 0.6 / graphs * graphnr
    return math.pow(10, factor)
def position_log(x, graphnr, graphs):
    position = x
    for i in range(0, graphnr):
        position += width_log(x, i, graphs)
    return position
def width_normal(x, graphnr, graphs):
    return x / graphs * 2
def position_normal(x, graphnr, graphs):
    position = x
    for i in range(0, graphnr):
        position += width_normal(x, i, graphs)
    return position

width_function = width_log if x_log else width_normal
position_function = position_log if x_log else position_normal

if not line_plot:
    def plot_graphs(graph_data, title, subtitle = "", index = 0):
        if index == len(sorted_indices) - 1:
            if len(subtitle) > 0: subtitle = '\n' + subtitle[:len(subtitle)-1]
            plot_axis = dict()
            graph_count = 0
            for m in range(0, measurements):
                if y_axis[m] != "PyAPI Time (s)": graph_count = graph_count + 1

            for m in range(0, measurements):
                if y_axis[m] == "PyAPI Time (s)": continue
                ax0 = plt.subplot(graph_count + 1, 1, m)
                plot_axis[y_axis[m]] = ax0

            for m in range(0, measurements):
                if y_axis[m] == "PyAPI Time (s)": continue
                ax0 = plot_axis[y_axis[m]]
                total_graphs = len(graph_data)
                j = 0
                for dictionary in graph_data:
                    x_data = sorted(dictionary.keys())
                    y_data = []
                    y_std = []
                    for x_value in x_data:
                        list.append(y_data, dictionary[x_value]['mean' + str(m)])
                        list.append(y_std, dictionary[x_value]['std' + str(m)])
                    ax0.bar([position_function(x,j,total_graphs) for x in x_data], y_data, bottom=bottom_value, width=[width_function(x,j,total_graphs) for x in x_data], yerr=y_std, label=file_titles[j], color=graph_colors[j], error_kw=dict(ecolor='black'))
                    j += 1
                if x_log: ax0.set_xscale('log')
                if y_log: ax0.set_yscale('log')
                if m == 1 or measurements == 1: ax0.set_title(title + subtitle)
                ax0.set_xlabel(x_axis)
                ax0.set_ylabel(y_axis[m])
                box = ax0.get_position()
                ax0.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                # Put a legend below current axis
                ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            for m in range(0, measurements):
                if y_axis[m] != "PyAPI Time (s)": continue
                ax0 = plot_axis['Total Time (s)']
                j = 0
                total_graphs = len(graph_data)
                for dictionary in graph_data:
                    x_data = sorted(dictionary.keys())
                    y_data = []
                    y_std = []
                    for x_value in x_data:
                        list.append(y_data, dictionary[x_value]['mean' + str(m)])
                        list.append(y_std, dictionary[x_value]['std' + str(m)])
                    ax0.bar([position_function(x,j,total_graphs) for x in x_data], y_data, bottom=bottom_value, width=[width_function(x,j,total_graphs) for x in x_data], color='0.75', alpha = 0.5)
                    j += 1

            if action.lower() == 'save':
                plt.savefig((title + subtitle + '.pdf').translate(None, '\n'), format="pdf")
                plt.close()
            else:
                plt.show()
        else:
            for key in graph_data[0].keys():
                new_graph_data = []
                for graph in graph_data:
                    list.append(new_graph_data, graph[key])
                plot_graphs(new_graph_data, title, subtitle + format_subtitle(axis[sorted_indices[index]], key) + ",",index + 1)
else:
    def plot_graphs(graph_data, title, subtitle = "", index = 0):
        if index == len(sorted_indices) - 1:
            if len(subtitle) > 0: subtitle = '\n' + subtitle[:len(subtitle)-1]
            plot_axis = dict()
            graph_count = 0
            for m in range(0, measurements):
                if y_axis[m] != "PyAPI Time (s)": graph_count = graph_count + 1
            for m in range(0, measurements):
                if y_axis[m] == "PyAPI Time (s)": continue
                ax0 = plt.subplot(graph_count + 1, 1, m)
                plot_axis[y_axis[m]] = ax0
            for m in range(0, measurements):
                if y_axis[m] != "PyAPI Time (s)": continue
                ax0 = plot_axis['Total Time (s)']
                j = 0
                for dictionary in graph_data:
                    x_data = sorted(dictionary.keys())
                    y_data = []
                    y_std = []
                    for x_value in x_data:
                        list.append(y_data, dictionary[x_value]['mean' + str(m)])
                        list.append(y_std, dictionary[x_value]['std' + str(m)])
                    ax0.plot(x_data, y_data, color=graph_colors[j])
                    ax0.fill_between(x=x_data, y1=y_data, y2=1e-6, color=graph_colors[j], alpha=0.5)
                    j += 1
            for m in range(0, measurements):
                if y_axis[m] == "PyAPI Time (s)": continue
                ax0 = plot_axis[y_axis[m]]
                j = 0
                for dictionary in graph_data:
                    x_data = sorted(dictionary.keys())
                    y_data = []
                    y_std = []
                    for x_value in x_data:
                        list.append(y_data, dictionary[x_value]['mean' + str(m)])
                        list.append(y_std, dictionary[x_value]['std' + str(m)])
                    ax0.errorbar(x_data, y_data, yerr=y_std, fmt='-o', label=file_titles[j], lw=2)
                    if y_axis[m] == 'Total Time (s)':
                        if fill_time:
                            ax0.fill_between(x=x_data, y1=y_data, y2=1e-6, color=graph_colors[j], alpha=0.2)
                    j += 1
                if x_log: ax0.set_xscale('log')
                if y_log: ax0.set_yscale('log')
                if m == 1 or measurements == 1: ax0.set_title(title + subtitle)
                ax0.set_xlabel(x_axis)
                ax0.set_ylabel(y_axis[m])
                box = ax0.get_position()
                ax0.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                # Put a legend below current axis
                ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if action.lower() == 'save':
                plt.savefig((title + subtitle + '.pdf').translate(None, '\n'), format="pdf")
                plt.close()
            else:
                plt.show()
        else:
            for key in graph_data[0].keys():
                new_graph_data = []
                for graph in graph_data:
                    list.append(new_graph_data, graph[key])
                plot_graphs(new_graph_data, title, subtitle + format_subtitle(axis[sorted_indices[index]], key) + ",",index + 1)
    # def plot_graphs(graph_data, title, subtitle = "", index = 0):
    #     if index == len(sorted_indices) - 1:
    #         if len(subtitle) > 0: subtitle = '\n' + subtitle[:len(subtitle)-1]
    #         for m in range(0, measurements):
    #             ax0 = plt.subplot(measurements, 1, m)
    #             j = 0
    #             for dictionary in graph_data:
    #                 x_data = sorted(dictionary.keys())
    #                 y_data = []
    #                 y_std = []
    #                 for x_value in x_data:
    #                     list.append(y_data, dictionary[x_value]['mean' + str(m)])
    #                     list.append(y_std, dictionary[x_value]['std' + str(m)])
    #                 ax0.errorbar(x_data, y_data, yerr=y_std, fmt='-o', label=file_titles[j])
    #                 j += 1
    #             if x_log: ax0.set_xscale('log')
    #             if y_log: ax0.set_yscale('log')
    #             if m == 1 or measurements == 1: ax0.set_title(title + subtitle)
    #             plt.xlabel(x_axis)
    #             plt.ylabel(y_axis[m])
    #             box = ax0.get_position()
    #             ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #             # Put a legend below current axis
    #             ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #         if action.lower() == 'save':
    #             plt.savefig((title + subtitle + '.pdf').translate(None, '\n'), format="pdf")
    #             plt.close()
    #         else:
    #             plt.show()
    #     else:
    #         for key in graph_data[0].keys():
    #             new_graph_data = []
    #             for graph in graph_data:
    #                 list.append(new_graph_data, graph[key])
    #             plot_graphs(new_graph_data, title, subtitle + format_subtitle(axis[sorted_indices[index]], key) + ",",index + 1)

plot_graphs(graph_data, graph_title)

