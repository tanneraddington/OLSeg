import cv2, os
import json

class Cell_Mask():
    '''
       This is the cell  mask class. It will contain a label, a bounding box,
       and all of the xy postions of the mask
       '''

    def __init__(self, label, xypos = [], bounding_box = []):
        '''
        Set edges to default to nothing.
        :param xpos: pixel location
        :param ypos: pixel location
        :param edges: what vertecies it connects to
        '''
        super(Vertex, xypos, bounding_box, label).__init__()
        self.xypos = xypos
        self.bounding_box = bounding_box
        self.edges = label

class Vertex():
    '''
    This is the vertex class, it contains its edges as well as its x and y postion
    '''
    def __init__(self, xpos, ypos, edges={}, marked=False, visited=False):
        '''
        Set edges to default to nothing.
        :param xpos: pixel location
        :param ypos: pixel location
        :param edges: what vertecies it connects to
        '''
        super(Vertex, xpos, ypos, edges, marked).__init__()
        self.xpos = xpos
        self.ypos = ypos
        self.edges = edges
        self.marked = marked
        self.visited = visited

    def add_edge(self, vertex):
        '''
        Add a vertex to the edge set
        :param vertex:
        :return:
        '''
        self.edges.append(vertex)

    def get_edges(self):
        '''
        getter for the edges
        :return:
        '''
        return self.edges




point_dict = dict()

def is_interest_point(image, i, j):
    '''
    This method checks to see if a pixel is part of the edge of a mask
    :param image:
    :param i:
    :param j:
    :return:
    '''
    if image[i][j] == 1:
        return image[i - 1] == 0 or image[i + 1] == 0 or image[j - 1] == 0 or image[j + 1]
    else:
        return False

def find_edges(image, vertex):
    '''
    This method finds creates an edge for each connected edge pixel
    :param image:
    :param vertex:
    :return:
    '''
    global point_dict
    x = vertex.xpos
    y = vertex.ypos
    for i in range(x-1, x+1):
        for j in range(y-1, y+1):
            if (i,j) in point_dict:
                vertex.add_edge(point_dict[(i,j)])
                continue
            if is_interest_point(image, i,j):
                iv = Vertex(i,j)
                point_dict[(i,j)] = iv
                vertex.add_edge(iv)
    return vertex


def make_graph(image,h,w):
    '''
    Input is a binary image. An Interest point is a point where it goes white to black or black to white.
    this function will loop through all of the pixels and set up the graph.
    :param image:
    :param h:
    :param w:
    :return:
    '''
    global point_dict
    # loop through each pixel in the image
    # we may need to add a white border to the image for this to work perfectly
    for i in range(1,h-1):
        for j in range(1,w-1):
            # if it is already in the dict of points, check to see if it has been marked, and
            # then if it has not been marked, find all the edges for that point.
            if (i,j) in point_dict:
                if point_dict[(i,j)].marked:
                    continue
                else:
                    iv = find_edges(image, point_dict[(i,j)])
                    iv.marked = True
                    point_dict[(i, j)] = iv
            if is_interest_point(image,i,j):
                interest_vertex = Vertex(i,j)
                iv = find_edges(image, interest_vertex)
                # after we find the edges mark that we have found the edges
                iv.marked = True
                point_dict[(i,j)] = iv
    return point_dict

def dfs(start_vertex):
    '''
    This will run a dfs to find a cycle, which is equal to a
    :param start_vertex:
    :return:
    '''
    global point_dict
    # add the spot to the bag
    mask_points = []
    bag = []
    bag.append(start_vertex)
    while (len(bag) > 0):
        # remove from bag
        vertex = bag.pop()
        if (not vertex.visited):
            # mark the vertex
            vertex.visited = True
            # add the point to the mask_points
            mask_points.append((vertex.xpos, vertex.ypos))
            # loop through each of the edges
            for edge in vertex.edges:
                bag.append(edge)

def find_masks(labels):
    '''
    This method drives the dfs, and loops through each
    :param labels:
    :return:
    '''
    global point_dict
    cur_masks = dict()
    key = 0
    for vertex in point_dict.items():
        if not vertex.visited:
            xypos = dfs(vertex)
            cur_masks[key] = xypos
            key = key + 1



def label_image(image, labels):
    '''
    This is the driver for creating json image labels. It generates the grpah, and then
    runs dfs on that graph to properly allocate
    :param image:
    :param labels:
    :return:
    '''
    global point_dict
    h = image.shape[0]
    w = image.shape[1]
    make_graph(image, h, w)
    cell_list = find_masks(labels)


def tiff_to_png(image_path):
    # convert image and return
    image = cv2.imread(image_path, 0)
    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 128
    # threshold the image
    img_binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_binary

def main():
    '''
    The main method takes user input for image path, and saves the json image in the same folder.
    :return:
    '''
    global point_dict
    # get the path to the data
    print("Batch or One Image?")
    num_vals = input()
    #TODO: setup for batch
    if(num_vals == "Batch" or num_vals == "batch"):
        print("NEED TO IMPLEMENT FOR BATCH")

    print("Input image path:")
    path = input()
    # change to black and white
    image = tiff_to_png(path)
    cv2.imshow("input image", image)
    print("NUMBER OF CELLS")
    cell_count = input().int()
    print("left to right top to bottom cell labels:")
    labels = []
    for cell in range(0,cell_count):
        print("cell #" + cell + ":")
        labels.append(input())

    label_image(image, labels)





if __name__ == '__main__':
    main()
