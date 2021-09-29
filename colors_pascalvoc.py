import numpy as np

class ColorMap_PASCALVOC:        
    # Person: person
    # Animal: bird, cat, cow, dog, horse, sheep
    # Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
    # Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
    # use the color here: /media/winstonfan/Workspace/Work/MyBuddy/models/pascal-voc-label-color-map.jpg
    BACKGROUND=[0,0,0]
    AEROPLANE=[14,10,10]
    BIKE=[13,20,14]
    BIRD=[50,50,0]
    BOAT=[1,0,50]
    BOTTLE=[51,0,50]
    BUS=[0,50,51]
    CAR=[75,0,0]
    CAT=[25,0,0]
    CHAIR=[50,50,50]
    COW=[25,50,0]
    DININGTABLE=[75,51,2]
    DOG=[26,0,50]
    HORSE=[76,0,51]
    MOTORBIKE=[25,50,50]
    PERSON=[75,50,50]
    POTTEDPLANT=[0,26,0]
    SHEEP=[50,25,0]
    SOFA=[0,75,0]
    TRAIN=[51,75,0]
    TVMONITOR=[0,25,52]
    VOID=[88,88,75]

    COLORS = []
    COLORS_BGR = []
    COLOR_MAP = {}

    # the sequence of colors in this arrar matters!!! as it maps to the prediction classes    
    COLORS.append(BACKGROUND)
    COLORS.append(AEROPLANE)
    COLORS.append(BIKE)
    COLORS.append(BIRD)
    COLORS.append(BOAT)
    COLORS.append(BOTTLE)
    COLORS.append(BUS)
    COLORS.append(CAR)
    COLORS.append(CAT)
    COLORS.append(CHAIR)
    COLORS.append(COW)
    COLORS.append(DININGTABLE)
    COLORS.append(DOG)
    COLORS.append(HORSE)
    COLORS.append(MOTORBIKE)
    COLORS.append(PERSON)
    COLORS.append(POTTEDPLANT)
    COLORS.append(SHEEP)
    COLORS.append(SOFA)
    COLORS.append(TRAIN)
    COLORS.append(TVMONITOR)
    COLORS.append(VOID)

    for color in COLORS:
        np_color = np.array(color)
        COLORS_BGR.append(np_color[[2,1,0]])

    COLOR_MAP['BACKGROUND'] = BACKGROUND
    COLOR_MAP['AEROPLANE'] = AEROPLANE
    COLOR_MAP['BIKE)'] = BIKE
    COLOR_MAP['BIRD'] = BIRD
    COLOR_MAP['BOAT'] = BOAT
    COLOR_MAP['BOTTLE'] = BOTTLE
    COLOR_MAP['BUS'] = BUS
    COLOR_MAP['CAR'] = CAR
    COLOR_MAP['CAT'] = CAT
    COLOR_MAP['CHAIR'] = CHAIR
    COLOR_MAP['COW'] = COW
    COLOR_MAP['DININGTABLE'] = DININGTABLE
    COLOR_MAP['DOG'] = DOG
    COLOR_MAP['HORSE'] = HORSE
    COLOR_MAP['MOTORBIKE'] = MOTORBIKE
    COLOR_MAP['PERSON'] = PERSON
    COLOR_MAP['POTTEDPLANT'] = POTTEDPLANT
    COLOR_MAP['SHEEP'] = SHEEP
    COLOR_MAP['SOFA'] = SOFA
    COLOR_MAP['TRAIN'] = TRAIN
    COLOR_MAP['TVMONITOR'] = TVMONITOR
    COLOR_MAP['VOID'] = VOID