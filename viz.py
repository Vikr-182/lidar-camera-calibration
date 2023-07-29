import json
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objs as go

arr = np.load("points.npy")
#x = (arr[0, ::1]/5).astype(np.int8)
#y = (arr[1, ::1]/5).astype(np.int8)
x = arr[0, ::1]
y = arr[1, ::1]
z = arr[2, ::1] * 0
trace = go.Scatter3d(
        x = x, y = y, z = z,mode = 'markers', marker = dict(size = 1)
)
trace2 = go.Scatter3d(
        x = [-200, 200], y = [-200, 200], z = [-200, 200], visible=True, marker=dict(size=0.00001)
)
layout = go.Layout(title = '3D Scatter plot')
fig = go.Figure(data = [trace, trace2], layout = layout)
fig.show()
