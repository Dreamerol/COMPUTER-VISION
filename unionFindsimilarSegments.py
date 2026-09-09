import numpy as np


class Node:
    def __init__(self, index, value):
        self.index = index
        self.value = value
        self.internal = 0

class Edge:
    def __init__(self, fro , to, w):
        self.fro = fro
        self.to = to
        self.w = w


class UnionFind:

    def __init__(self, size, nodes):
        self.parent = [i for i in range(size)]
        self.sizes = np.ones(size)
        self.nodes = nodes

    def find(self, a):
        if self.parent[a] != a:
            return self.find(self.parent[a]) # we call a class method with self
        else:
            return a
        
    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)

        if(self.sizes[pa] > self.sizes[pb]):
            self.parent[pb] = pa
            self.sizes[pa] += self.sizes[pb] 
            self.nodes[pa].internal += self.nodes[pb].internal
            self.nodes[pb].internal = 0

        else:
            self.parent[pa] = pb
            self.sizes[pb] += self.sizes[pa] 
            self.nodes[pb].internal += self.nodes[pa].internal
            self.nodes[pa].internal = 0


image = [[1, 2, 3],
         [4, 3, 2],
         [5, 6, 7]]

size = 3

image_flat = image.flatten()

nodes = []
edges = []

for i in range(len(image_flat)):
    node = Node(i, image_flat[i])
    nodes.append(node)


def valid(x, y):
    if x < 0 or x >= size or y < 0 or y >= size:
        return False
    return True
def find_coordinates(x, y):
    nbrs = []
    vec = [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]
    
    for item in vec:
        if valid(item[0], item[1]):
            nbrs.append(item)
    return nbrs

edges = []
connect = set()

for i in range(size):
    for j in range(size):
        nbrs = find_coordinates(i, j)
        if (i, j) in connect or (j, i) in connect:
            continue
        connect.add((i, j))
        edgi = Edge(i, j, image_flat[i*size + j])
        edges.append(edgi)

edges = edges.sort(lambda k: k.w)
un = UnionFind(nodes)
k = 4

for edge in edges:
    fr = edge.fro
    to = edge.to
    w = edge.w
    if w <= min(un.nodes[fr].internal + k/un.nodes[fr].internal, un.nodes[to].internal + k/un.nodes[to].internal):
        un.union(fr, to)



def form_region(pixels):
    xs = []
    ys = []
    for p in pixels:
        x = p % size
        y = p // size

        xs.append(x)
        ys.append(y)
    return [min(xs), max(xs), min(ys), max(ys)]

pixels = {}

for i in range(len(un.parent)):
    pixels[un.parent[i]].append(i)

regions = []

for key,value in pixels:
    print(form_region(value))
    regions.append(form_region(value))

def is_dot_inside(dot, region):
    x, y = dot
    xmin,xmax,ymin, ymax = region
    if x <= xmax and x >= xmin and y <= ymax and y >= ymin:
        return True
    return False


def are_nbrs(region_one, region_two):
    for dot in region_one:
        if(is_dot_inside(dot, region_one)):
            return True

    return False


def make_color_hist(region):
    bins = {}

    bins[0] = 0
    bins[63] = 0
    bins[128] = 0
    bins[167] = 0

    x1,x2,y1,y2 = region
    for i in range(x1, x2 + 1, 1):
        for j in range(y1, y2 + 1, 1):
            if image[i][j] >= 167:
                bins[167] += 1
            elif image[i][j] >= 128:
                bins[128] += 1

            elif image[i][j] >= 63:
                bins[63] += 1
            else:
                bins[0] += 1


def compute_possible_dots(dot):
    [x, y] = dot
    nbrs = []
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            if valid(x+i, j+y):
                nbrs.append([x+i, j+y])
    return nbrs

def make_texture_diagram(region):
    xmin,xmax,ymin, ymax = region
    bins = {}
    for i in range(0, 255, 1):
        bins[i] = 0
    for i in range(xmin, xmax + 1, 1):
        for j in range(ymin, ymax + 1, 1):
            dots = compute_possible_dots(image[i][j])
            for d in dots:
                sum = 0
                for l in range(len(d)):
                    sum += 2**(7-l)*(image[i][j] < image[d[0]][d[1]])
                bins[sum] += 1
    return bins


def to_combine(regions):
    if len(regions) < 5:
        return regions    
    new_regions = []
    for i in range(len(regions)):
        for j in range(i+1, len(regions), 1):
            if are_nbrs(regions[i], regions[j]):
                bins_one = make_color_hist(regions[i])
                bins_two = make_color_hist(regions[j])

                bins_three = make_texture_diagram(regions[i])
                bins_four = make_texture_diagram(regions[j])

                probs_color = 0
                probs_text = 0

                for i in range(len(bins_one)):
                    probs_color += min(bins_one[i], bins_two[i])/max(bins_one[i], bins_two[i])
                for j in range(len(bins_three)):
                    probs_text += min(bins_three[i], bins_four[i])/max(bins_three[i], bins_four[i])
                    
                if probs_color >= 0.7 and probs_text >= 0.7:
                    new_reg = [min(regions[i][0], regions[j][0]), max(regions[i][1], regions[j][1]), min(regions[i][2], regions[j][2]), max(regions[i][3], regions[j][3])]
                    new_regions.append(new_reg)
                else:
                    new_regions.append(regions[i])
                    new_regions.append(regions[j])

    return to_combine(new_regions)

# CNN we get image[region] - we get the pixels within the region -> extract them to feature vectors -> we pass them to CNN -> SVM them   
  
class SVM:
    def __init__(self, learning_rate, n_features):
        self.learning_rate = learning_rate
        self.w = np.random.randn(n_features)
        self.b = 0.4

    def forward(self, x, y):
        return y*(x @ self.w.T + self.b)

    def train(self, x, y):
        out = self.forward(x)
        mask = out < 1

        dw = y[mask, None]*x
        db = -np.sum(y[mask], axis=0)

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

# y[mask, None] - adds another dimension


