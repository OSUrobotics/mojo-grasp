#! /usr/bin/env python3

import json
import os


# directory = os.path.dirname(__file__)
directory = os.getcwd()

def test():
    name = input("hand_name:  ")
    hand = {
        "hand_name": name,
        "palm": {
            "left_joint": input("palm left_joint (pin or spring):  "),
            "right_joint": input("palm right_joint (pin or spring):  "),
            "stiffness": 0
        },
        "left_finger": {
            "number_segments": int(input("left finger number of segments(1, 2, or 3):  ")),
            "finger_length": float(input("total length of the left finger(default=0.12m):  ")),
            "segment_names": list(input("list of segment names(ex: proximal intermediate distal):  ").split(" ")),
            "proximal": {
                "ratio": float(input("left proximal ratio: \n(if finger only has one segment enter 0) \n ")),
                "joint_bottom": input("left proximal bottom joint (pin or spring): \n(same as palm left_joint) \n  "),
                "joint_top": input("left proximal top joint (pin or spring):  ")
            },
            "intermediate": {
                "ratio": float(input("left intermediate ratio:\n(if finger is one or two segments enter 0) \n  ")),
                "joint_bottom": input("left intermediate bottom joint (pin or spring): \n(Same as right proximal top joint)\n(if finger is one or two segments enter NA) \n  "),
                "joint_top": input("left intermediate top joint (pin or spring): \n(if finger is one or two segments enter NA) \n  ")
            },
            "distal": {
                "ratio": float(input("left distal ratio:  ")),
                "joint_bottom": input("left distal bottom joint(pin or spring): \n(depending on # of segments either same as proximal or intermediate top joint)\n  "),
                "ending": input("left distal ending (round):  "),
                "sensors": {
                    "num": int(input("number of sensors (0-9): ")),
                    "type": list(input("list the types of sensors in order: \n(as of now the name does not do anything but in the furute will be used for the type of sensor) \n ").split(" "))
                }
            }
        },
        "right_finger": {
            "number_segments": int(input("right finger number of segments(1, 2, or 3):  ")),
            "finger_length": float(input("total length of the right finger(default=0.12m):  ")),
            "segment_names": list(input("list of segment names(ex: proximal intermediate distal):  ").split(" ")),
            "proximal": {
                "ratio": float(input("right proximal ratio: \n(If finger onle has one segment enter 0) \n  ")),
                "joint_bottom": input("right proximal bottom joint (pin or spring): \n(same as palm right_joint) \n  "),
                "joint_top": input("right proximal top joint:  ")
            },
            "intermediate": {
                "ratio": float(input("right intermediate ratio: \n(if finger is one or two segments enter 0) \n  ")),
                "joint_bottom": input("right intermediate bottom joint (pin or spring): \n(Same as right proximal top joint) \n(if finger is one or two segments enter NA) \n  "),
                "joint_top": input("right intermediate top joint (pin or spring): \n(if finger is one or two segments enter NA) \n  ")
            },
            "distal": {
                "ratio": float(input("right distal ratio:  ")),
                "joint_bottom": input("right distal bottom joint(pin or spring): \n(depending on # of segments either same as proximal or intermediate top joint)\n  "),
                "ending": input("right distal ending (round):  "),
                "sensors": {
                    "num": int(input("number of sensors (0-9):  ")),
                    "type": list(input("list the types of sensors in order: \n(as of now the name does not do anything but in the furute will be used for the type of sensor) \n  ").split(" "))
                }
            }
        },#in meters
        "hand_parameters": {
            "palm_width": float(input("palm width(default=0.14m):  ")),
            "palm_height": float(input("palm height(default=0.04m):  ")),
            "hand_thickness": float(input("hand thickness(default=0.02m):  ")),
            "finger_width": float(input("finger width(default=0.02m):  ")),
            "finger_length": float(input("finger length(default=0.12m):  ")),
            "joint_length": float(input("joint length(default=0.01125m):  ")),
            "scale_factor": float(input("scale factor(default=1):  "))
        }
    }
    print(json.dumps(hand, indent=4))

    print(f'{directory}/hand_models/hand_queue_json/')
    with open(f'{directory}/hand_models/hand_queue_json/{name}.json', 'w') as file:
        json.dump(hand, file, indent=4)


if __name__ == '__main__':

    # hand_file()
    test()