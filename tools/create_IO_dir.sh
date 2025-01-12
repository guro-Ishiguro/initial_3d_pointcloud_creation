#!/bin/bash

read -p "Input dir name: " dir_name

mkdir -p ../data/$dir_name/{images/{drone,disparity,depth},txt} \
         ../output/$dir_name/{point_cloud,video,mesh}