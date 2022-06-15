import cv2, os
import json

class Cell_Mask():
    '''
       This is the cell  mask class. It will contain a label, a bounding box,
       and all of the xy postions of the mask
       '''

    def __init__(self, label = ""):
        '''
        Set edges to default to nothing.
        :param xpos: pixel location
        :param ypos: pixel location
        :param edges: what vertecies it connects to
        '''
        self.xposes = set()
        self.yposes = set()
        self.bounding_box = set()
        self.label = label
        self.sum = -1

    def calculate_bb(self):
        # make sure this does not ex
        self.bounding_box.add(min(self.xposes))
        self.bounding_box.add(min(self.yposes))
        self.bounding_box.add(max(self.xposes))
        self.bounding_box.add(max(self.yposes))


    def inside_box(self, bounding_box2):
        '''
        This method make sure if two cells are within one another they are the same cell and are added as xypositions
        in the list.
        :param bounding_box2:
        :return:
        '''
        inside = False
        self_bb = list(self.bounding_box)
        bounding_box2 = list(bounding_box2)
        for index in range(0,1):
            if self_bb[index] < bounding_box2[index] and self_bb[index + 2] > bounding_box2[index + 2]:
                inside = True
            else:
                return False
        return inside


    def print_mask(self):
        '''
        This method prints the masks.
        :return:
        '''
        print("MASK LENGTH: " + str(len(self.xposes)))
        print("X Y POSITIONS:")
        xpos = list(self.xposes)
        ypos = list(self.yposes)
        print(xpos)
        print(ypos)
        # for index in range(0, len(xpos)):
        #     print(str(xpos[index]) + ":" + str(ypos[index]))
        print("Label:" + self.label)
        print("Bounding Box:" + str(self.bounding_box))

class Vertex():
    '''
    This is the vertex class, it contains its edges as well as its x and y postion
    '''
    def __init__(self, xpos, ypos):
        '''
        Set edges to default to nothing.
        :param xpos: pixel location
        :param ypos: pixel location
        :param edges: what vertecies it connects to
        '''
        self.xpos = xpos
        self.ypos = ypos
        self.edges = set()
        self.marked = False
        self.visited = False

    def add_edge(self, vertex):
        '''
        Add a vertex to the edge set
        :param vertex:
        :return:
        '''
        self.edges.add(vertex)



def is_interest_point(image, i, j):
    '''
    This method checks to see if a pixel is part of the edge of a mask
    :param image:
    :param i:
    :param j:
    :return:
    '''
    if image[i][j] == 0:
        if(image[i][j - 1] == 255 or image[i][j + 1] == 255 or image[i + 1][j] == 255 or image[i - 1][j] == 255):
            return True
    else:
        return False

def find_edges(image, i, j , point_dict):
    '''
    This method finds creates an edge for each connected edge pixel
    :param image:
    :param vertex:
    :return:
    '''
    vertex = Vertex(i, j)
    x = vertex.xpos
    y = vertex.ypos
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if i == x and j == y:
                continue
            key = str(i) + ":" + str(j)
            if key in point_dict:
                vertex.add_edge(point_dict[key])
                continue
            if is_interest_point(image, i,j):
                iv = Vertex(i,j)
                point_dict[key] = iv
                vertex.add_edge(iv)
    return vertex, point_dict


def make_graph(image,h,w, point_dict):
    '''
    Input is a binary image. An Interest point is a point where it goes white to black or black to white.
    this function will loop through all of the pixels and set up the graph.
    :param image:
    :param h:
    :param w:
    :return:
    '''
    # loop through each pixel in the image
    # we may need to add a white border to the image for this to work perfectly
    for i in range(1,h-1):
        for j in range(1,w-1):
            # if it is already in the dict of points, check to see if it has been marked, and
            # then if it has not been marked, find all the edges for that point.
            key = str(i) + ":" + str(j)
            if key in point_dict:
                if point_dict[key].marked:
                    continue
                else:
                    iv, point_dict = find_edges(image, i,j, point_dict)
                    iv.marked = True
                    point_dict[key] = iv
            if is_interest_point(image,i,j):
                iv, point_dict = find_edges(image, i,j, point_dict)
                # after we find the edges mark that we have found the edges
                iv.marked = True
                key = str(i) + ":" + str(j)
                point_dict[key] = iv

    return point_dict

def dfs(start_vertex, point_dict):
    '''
    This will run a dfs to find a cycle, which is equal to a
    :param start_vertex:
    :return:
    '''
    # add the spot to the bag
    cur_mask = Cell_Mask()
    bag = []
    start_key = str(start_vertex.xpos) + ":" + str(start_vertex.ypos)
    bag.append(start_key)
    while (len(bag) > 0):
        # remove from bag
        vertex_key = bag.pop()
        if (not point_dict[vertex_key].visited):
            # mark the vertex
            point_dict[vertex_key].visited = True
            # add the point to the mask_points
            cur_mask.xposes.add(point_dict[vertex_key].xpos)
            cur_mask.yposes.add(point_dict[vertex_key].ypos)
            # loop through each of the edges
            for edge in point_dict[vertex_key].edges:
                edge_key = str(edge.xpos) + ":" + str(edge.ypos)
                bag.append(edge_key)

    return cur_mask, point_dict

def find_masks(labels, point_dict):
    '''
    This method drives the dfs, and loops through each
    :param labels:
    :return:
    '''
    cur_masks = dict()
    key_val = 0
    for key in point_dict.keys():
        vertex = point_dict[key]
        if not vertex.visited:
            print(vertex)
            mask, point_dict = dfs(vertex, point_dict)
            cur_masks[key_val] = mask
            key_val = key_val + 1
    masks = []
    for key in cur_masks.keys():
        cur_masks[key].calculate_bb()
        cur_masks[key].sum = sum(cur_masks[key].bounding_box)
        masks.append(cur_masks[key])

    # here we will check all the bounding boxes
    if len(cur_masks) > len(labels):
        print("UNEVEN")

    masks.sort(key= lambda x:x.sum)
    # set the labels
    for index in range(0, len(labels)):
        masks[index].label = labels[index]

    for mask in masks:
        mask.print_mask()

    return masks



def label_image(image, labels):
    '''
    This is the driver for creating json image labels. It generates the grpah, and then
    runs dfs on that graph to properly allocate
    :param image:
    :param labels:
    :return:
    '''
    point_dict = dict()
    h = image.shape[0]
    w = image.shape[1]
    point_dict = make_graph(image, h, w, point_dict)
    cell_list = find_masks(labels, point_dict)
    print(cell_list)
    # add to json


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
    cell_count = int(input())
    print("left to right top to bottom cell labels:")
    labels = []
    for cell in range(0,cell_count):
        print("cell #" + str(cell) + ":")
        labels.append(input())

    label_image(image, labels)





if __name__ == '__main__':
    main()
