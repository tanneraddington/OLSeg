import cv2, os
import json

class Cell_Mask():
    '''
       This is the cell  mask class. It will contain a label, a bounding box,
       and all of the xy postions of the mask
       '''

    def __init__(self, label = "Cell"):
        '''
        Set edges to default to nothing.
        :param xpos: pixel location
        :param ypos: pixel location
        :param edges: what vertecies it connects to
        '''
        self.xposes = []
        self.yposes = []
        self.xmax = -1
        self.ymax = -1
        self.xmin = -1
        self.ymin = -1
        self.bounding_box = set()
        self.label = label
        self.sum = -1
        self.cell_num = -1
        self.area = 0
        self.marked = False

    def calculate_bb(self):
        # make sure this does not ex
        self.xmax = max(self.xposes)
        self.ymax = max(self.yposes)
        self.xmin = min(self.xposes)
        self.ymin = min(self.yposes)
        self.bounding_box.add(min(self.xposes))
        self.bounding_box.add(min(self.yposes))
        self.bounding_box.add(max(self.xposes))
        self.bounding_box.add(max(self.yposes))
        self.area = (self.ymax - self.ymin) * (self.xmax - self.xmin)


    def inside_box(self, xmax, ymax, xmin, ymin):
        '''
        This method make sure if two cells are within one another they are the same cell and are added as xypositions
        in the list.
        :param bounding_box2:
        :return:
        '''
        if self.xmin > xmin and self.ymin > ymin and self.xmax < xmax and self.ymax < ymax :
            return True
        else:
            return False


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
        print("YMIN: " + str(self.xmin))
        print("YMAX: " + str(self.xmax))
        print("XMIN: " + str(self.ymin))
        print("XMAX: " + str(self.ymax))
        print("AREA: " + str(self.area))

    def create_json_dict(self):
        json_dict = dict()
        list_of_pos = [None] * len(self.xposes)
        for index in range(0, len(self.xposes)):
            list_of_pos[index] = (self.yposes[index], self.xposes[index])
        x_y = list_of_pos
        json_dict["points"] = x_y
        json_dict["label"] = self.label

        # look at best way to make this json
        return json_dict

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
            cur_mask.xposes.append(point_dict[vertex_key].xpos)
            cur_mask.yposes.append(point_dict[vertex_key].ypos)
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
            mask, point_dict = dfs(vertex, point_dict)
            cur_masks[key_val] = mask
            key_val = key_val + 1
    masks = []

    for key in cur_masks.keys():
        cur_masks[key].calculate_bb()
        cur_masks[key].sum = sum(cur_masks[key].bounding_box)
        masks.append(cur_masks[key])

    masks.sort(key= lambda x:x.area, reverse=False)

    final_masks = []
    # new algo for seeing if boxes are contained
    while(len(masks) > 0):
        top_mask = masks.pop()
        if top_mask.marked:
            continue
        # check each remaining mask combo to see if it is in the box
        for mask in masks:
            if mask.inside_box(top_mask.xmax, top_mask.ymax, top_mask.xmin, top_mask.ymin):
                top_mask.xposes = top_mask.xposes + mask.xposes
                top_mask.yposes = top_mask.yposes + mask.yposes
                mask.marked = True

        final_masks.append(top_mask)

    print("ORIG")
    for mask in final_masks:
        mask.print_mask()

    return final_masks



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
    cell_dict = dict()
    index = 0
    shapes = []
    for mask in cell_list:
        mask_dict = mask.create_json_dict()
        shapes.append(mask_dict)
    cell_dict["shapes"] = shapes
    return cell_dict
    # add to json


def tiff_to_png(image_path):
    # convert image and return
    image = cv2.imread(image_path, 0)
    cv2.imshow("input image", image)
    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 128
    # threshold the image
    img_binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_binary

def make_json(path, json_pth):
    '''
    This method creates json for 1 image. It is used in main to create for batches as well.
    :param path:
    :return:
    '''
    # print("Input image path:")
    # path = input()
    # change to black and white
    image = tiff_to_png(path)
    # print("NUMBER OF CELLS")
    # cell_count = int(input())

    # print("left to right top to bottom cell labels:")
    labels = []
    # for cell in range(0,cell_count):
    #     print("cell #" + str(cell) + ":")
    #     labels.append(input())

    cell_dict = label_image(image, labels)
    cell_dict["imagePath"] = path.replace('_mask', '').replace('.tif', '.png')
    cell_dict["imageHeight"] = image.shape[0]
    cell_dict["imageWidth"] = image.shape[1]
    # write the json
    with open(json_pth, "w") as write:
        json.dump(cell_dict, write)

def main():
    '''
    The main method takes user input for image path, and saves the json image in the same folder.
    :return:
    '''

    # get the path to the data
    # print("Batch or One Image?")
    # num_vals = input()

    ### CHANGE DIR PATH HERE ###
    dir_path = "/Users/tannerwatts/Desktop/OLSeg/detectron_segmentation/test"

    # loop through each image in a directory.
    image_list = []
    index = 1
    for filename in [file for file in os.listdir(dir_path)]:
        image_pth = os.path.join(dir_path, filename)
        if image_pth.__contains__('mask') and not image_pth.__contains__('.json'):
            print("IMAGE #: " + str(index))
            print(filename)
            image_name = filename.replace('.tif','.json')
            json_pth = os.path.join(dir_path, image_name)
            make_json(image_pth,json_pth)
            index = index + 1



if __name__ == '__main__':
    main()
