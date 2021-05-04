#! /usr/bin/env python3

import json
import os


directory = os.path.dirname(__file__)


def test():
    name = input("hand_name:  ")
    hand = {
        "hand_name": name,
        "palm": {
            "left_joint": input("palm left_joint:  "),
            "right_joint": input("palm right_joint:  "),
            "stiffness": 0
        },
        "left_finger": {
            "number_segments": int(input("left finger number of segments(default=3):  ")),
            "finger_length": float(input("total length of the left finger(default=0.12):  ")),
            "segment_names": list(input("list of seg names:  ").split(" ")),
            "prox": {
                "ratio": float(input("left prox ratio:  ")),
                "joint_bottom": input("left prox bottom joint:  "),
                "joint_top": input("left prox top joint:  ")
            },
            "intermediate": {
                "ratio": float(input("left intermediate ratio:  ")),
                "joint_bottom": input("left intermediate bottom joint:  "),
                "joint_top": input("left intermediate top joint:  ")
            },
            "distal": {
                "ratio": float(input("left distal ratio:  ")),
                "joint_bottom": input("left distal bottom joint:  "),
                "ending": input("left distal ending:  "),
                "sensors": {
                    "num": int(input("number of sensors: ")),
                    "type": list(input("list the types of sensors in order:  ").split(" "))
                }
            }
        },
        "right_finger": {
            "number_segments": int(input("right finger number of segments:  ")),
            "finger_length": float(input("total length of the right finger:  ")),
            "segment_names": list(input("list of seg names:  ").split(" ")),
            "prox": {
                "ratio": float(input("right prox ratio:  ")),
                "joint_bottom": input("right prox bottom joint:  "),
                "joint_top": input("right prox top joint:  ")
            },
            "intermediate": {
                "ratio": float(input("right intermediate ratio:  ")),
                "joint_bottom": input("right intermediate bottom joint:  "),
                "joint_top": input("right intermediate top joint:  ")
            },
            "distal": {
                "ratio": float(input("right distal ratio:  ")),
                "joint_bottom": input("right distal bottom joint:  "),
                "ending": input("right distal ending:  "),
                "sensors": {
                    "num": int(input("number of sensors:  ")),
                    "type": list(input("list the types of sensors in order:  ").split(" "))
                }
            }
        },#in meters
        "hand_parameters": {
            "palm_width": .14,
            "palm_height": 0.04,
            "hand_thickness": .02,
            "finger_width": .02,
            "finger_length": .12,
            "joint_length": .01125,
            "scale_factor": 1
        }
    }
    print(json.dumps(hand, indent=4))

    with open(f'{directory}/hand_models/hand_queue_json/{name}.json', 'w') as file:
        json.dump(hand, file, indent=4)


if __name__ == '__main__':

    # hand_file()
    test()