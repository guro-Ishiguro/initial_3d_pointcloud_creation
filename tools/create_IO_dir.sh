#!/bin/bash

read -p "Input dir name: " dir_name

mkdir -p ../data/$dir_name/{images,txt,videos} \
         ../output/$dir_name/{point_cloud,video,mesh,disparity,depth}