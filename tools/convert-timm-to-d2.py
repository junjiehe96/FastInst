#!/usr/bin/env python

import pickle as pkl
import sys

import torch

"""
Usage:
  # run the conversion
  ./convert-timm-to-d2.py resnet50d_ra2-464e36ba.pth resnet50d_ra2-464e36ba.pkl
  # Then, use resnet50d_ra2-464e36ba.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/resnet50d_ra2-464e36ba.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":

    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        k = "backbone." + k
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "timm", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
