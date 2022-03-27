import os
import sys
import numpy as np
import matplotlib.pyplot as pyplot

colors = {'ceiling':[0,255,0],
          'floor':[0,0,255],
          'wall':[0,255,255],
          'beam':[255,255,0],
          'column':[255,0,255],
          'window':[100,100,255],
          'door':[200,200,100],
          'table':[170,120,200],
          'chair':[255,0,0],
          'sofa':[200,100,100],
          'bookcase':[10,200,100],
          'board':[200,200,200],
          'clutter':[50,50,50]}
colors = list(colors.values())

colors2 = [[50,50,50]]

colors7 = [[255, 0, 0], [255, 125, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255]]

colors72 = [[242,183,176], [183,205,225], [210,234,200], [219,204,226], [249,218,173], [255,255,209], [227,216,192]]

colors40 = [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]]

def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        #c = colors[labels[i]]
        #c = [int(x * 255) for x in c]
        c = colors[labels[i]]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def write_ply_rgb(points, rgb, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        #c = colors[labels[i]]
        #c = [int(x * 255) for x in c]
        c = rgb[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def write_ply_color_modelnet40(points, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    #labels = labels.astype(int)
    N = points.shape[0]
    #if num_classes is None:
    #    num_classes = np.max(labels) + 1
    #else:
    #    assert (num_classes > np.max(labels))
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        #c = colors[labels[i]]
        #c = [int(x * 255) for x in c]
        c = colors2[0]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def write_ply_color_shapenet(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        #c = colors[labels[i]]
        #c = [int(x * 255) for x in c]
        c = colors7[labels[i]%7]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()