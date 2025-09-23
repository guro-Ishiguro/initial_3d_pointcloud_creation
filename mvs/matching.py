import sys
import numpy
import os


# 時刻をキーとするリストの生成
def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    parsed_list = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    parsed_list = [(float(l[0]), l[1:]) for l in parsed_list if len(l) > 1]
    return dict(parsed_list)


# 2 つのファイルの時刻のマッチング
def associate(first_list, second_list, offset, max_difference):
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, first_list[a], second_list[b]))
    matches.sort()
    return matches
