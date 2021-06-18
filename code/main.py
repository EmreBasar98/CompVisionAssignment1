import numpy as np
import math
import cv2
from math import pi
import os
from xml.etree import ElementTree as et
import sys

def readXmlandCalculate(input,area,i1,i2 ):

    file_name = input + ".xml"
    full_file = os.path.join('annotations', file_name)

    root = et.parse(full_file).getroot()
    xmin = int(root.findall('object/bndbox/xmin')[0].text)
    ymin = int(root.findall('object/bndbox/ymin')[0].text)
    xmax = int(root.findall('object/bndbox/xmax')[0].text)
    ymax = int(root.findall('object/bndbox/ymax')[0].text)

    actual_area = (xmax - xmin) * (ymax - ymin)

    overlap1_x = max(xmin,i1[0])
    overlap1_y = max(ymin,i1[1])

    overlap2_x = min(xmax,i2[0])
    overlap2_y = min(ymax,i2[1])


    overlap_area = abs((overlap2_x - overlap1_x)) * abs((overlap2_y - overlap1_y))

    if(overlap_area > actual_area):
        overlap_area = 0

    result = overlap_area / (area + actual_area - overlap_area)
    if(result < 0):
        result = 0
    print("IOU = ", result)

def findLine(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x),int(y)



def houghTransform(edge):
    theta = np.zeros(shape=(180,),dtype=int)

    for i in range(180):
        theta[i] = i

    cos = np.cos(theta*(pi/180))
    sin = np.sin(theta*(pi/180))

    rho_range = round(math.sqrt(edge.shape[0]**2 + edge.shape[1]**2))
    accumulator = np.zeros((2 * rho_range, len(theta)), dtype=np.uint8)


    peak_values = np.where(edge == 255)
    coordinates = list(zip(peak_values[0], peak_values[1]))


    for p in range(len(coordinates)):
        for t in range(len(theta)):
            rho = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[rho, t] += 1 # Suppose add 1 only, Just want to get clear result

    return accumulator

# -------------------------- main -------------------------- #
if __name__ == '__main__':

    input = sys.argv[1]

    image = cv2.imread("images/"+input+".png")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale,75,150)

    accumulator = houghTransform(edges)

    height, width = edges.shape

    coordinates = []
    for i in range(0,len(accumulator)):
        for j in range(0,len(accumulator[i])):
            if( j in range(0,5) or j  in range(175,180)):
                if(accumulator[i][j] > 40):
                    coordinates.append((i,j))
            if (j in range(85, 95) ):
                if (accumulator[i][j] > 100):
                    coordinates.append((i, j))
    if(len(coordinates) == 0):
        print("no line detected")
        sys.exit()

    final_coordinates = []
    final_coordinates.append(coordinates[0])
    for i in coordinates:
        addable = True
        theta = i[1]
        if(theta in range(0,10) or theta  in range(170,180) or theta  in range(85,95)):
            iter = len(final_coordinates)
            for j in range(0, iter):
                elm = final_coordinates[j]
                elm_rho = elm[0]
                if(abs(theta - elm[1]) > 70 ):
                    pass
                elif((i[0] - 30 <= elm_rho <= i[0]+ 30)):
                    addable = False
            if(addable):
                final_coordinates.append(i)

    recVer = []
    recHor = []

    for i in range(0, 3):
        final_coordinates.pop()

    for i in final_coordinates:
        rho = i[0]
        theta = i[1]
        if(theta in range(75, 105)):
            recHor.append(i)
        if(theta in range(0,10) or theta  in range(170,180)):
            recVer.append(i)


    line1 = None
    line2 = None
    line3 = None
    line4 = None

    if (len(recVer) == 2):
        line1 = recVer[0]
        line2 = recVer[1]
    elif (len(recVer) > 2):
        line1 = recVer[len(recVer)//2 -3 ]
        line2 = recVer[len(recVer)//2 -1 ]

    if (len(recHor) == 2):
        line3 = recHor[0]
        line4 = recHor[1]
    elif (len(recHor) > 2):
        line3 = recHor[len(recHor) //2  ]
        line4 = recHor[(len(recHor) // 2) + 1]

    recCoord = []
    if(line1 is not None):
        recCoord.append(line1)

    if(line2 is not None):
        recCoord.append(line2)

    if(line3 is not None):
        recCoord.append(line3)

    if(line4 is not None):
        recCoord.append(line4)

    lines = []

    for i in range(0, len(recCoord)):
        rho = recCoord[i][0]
        theta = recCoord[i][1]
        a = np.cos(theta*(pi/180))
        b = np.sin(theta*(pi/180))
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        line = findLine( (x1,y1),(x2,y2))
        lines.append(line)

    if(len(lines) == 4):
        i1 = intersection(lines[0],lines[2])
        i2 = intersection(lines[1],lines[3])

        cv2.rectangle(image, i2 ,i1,(255,0,0),3)

        area = abs( i2[0] - i1[0]) * abs( i2[1]- i1[1] )
        readXmlandCalculate(input, area, i1, i2)

    o = input+"out.jpg"
    e = input+"edge.jpg"

    cv2.imwrite(o,image)
    cv2.imwrite(e,edges)
