import json


import os


import csv

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pycountry_convert as pc

from pprint import pprint
from typing import Tuple

from tqdm import tqdm

import time
from torchvision.transforms.functional import rotate
from PIL import Image, ImageFile
from torchvision import transforms
tqdm.pandas()


# filter
# count = 0
# list = []
# for j in range(0,18):
#     with open('/mnt/data/home/kxj200023/data/yfcc100m/images_minbyte_10_valid_uploaded_date/' + str(j) + '/metadata.json') as f:
#         meta = json.load(f)
#         for key in range(len(meta)):
#             one_meta = meta[key]
#             filename = one_meta["ID"] + "." + one_meta["EXT"]
#             a = os.path.exists("/mnt/data/home/kxj200023/data/yfcc100m/images_minbyte_10_valid_uploaded_date/" + str(j) + "/images/" + filename)
#             b = one_meta["LON"] != ""
#             c = one_meta["LAT"] != ""
#             d = (one_meta["EXT"]=="jpg" or one_meta["EXT"]=="png")
#             # e = ("indoor" in one_meta["AUTO_TAG_SCORES"] or "outdoor" in one_meta["AUTO_TAG_SCORES"])
#
#             if a and b and c and d:
#                 one_meta["path"] = "/home/kxj200023/data/yfcc100m/images_minbyte_10_valid_uploaded_date/" + str(j) + "/images/" + filename
#                 list.append(one_meta)
#                 count = count + 1
#
# with open('/mnt/data/home/kxj200023/data/yfcc100m/'+'0-17_filtered.json', 'w') as fp:
#     json.dump(list, fp)
#
#
#
# # preprocessing
# def get_continent_name(continent_code: str) -> str:
#     continent_dict = {
#         "NA": "North America",
#         "SA": "South America",
#         "AS": "Asia",
#         "AF": "Africa",
#         "OC": "Oceania",
#         "EU": "Europe",
#         "AQ" : "Antarctica"
#     }
#     return continent_dict[continent_code]
#
# def get_continent(lat: float, lon: float) -> Tuple[str, str]:
#     geolocator = Nominatim(user_agent="<username>@gmail.com", timeout=10)
#     geocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.01)
#
#     location = geocode(f"{lat}, {lon}", language="en")
#
#     # for cases where the location is not found, coordinates are antarctica
#     if location is None:
#         return "Antarctica"
#
#     # extract country code
#     address = location.raw["address"]
#     country_code = address["country_code"].upper()
#
#     # get continent code from country code
#     continent_code = pc.country_alpha2_to_continent_code(country_code)
#     continent_name = get_continent_name(continent_code)
#
#     return continent_name
#
# new_list = []
#
# with open('/mnt/data/home/kxj200023/data/yfcc100m/'+'0-17_filtered.json') as f:
#     time1 = time.time()
#     list = json.load(f)
#     for i in range(len(list)):
#         meta = list[i]
#         lon = float(meta["LON"])
#         lat = float(meta["LAT"])
#
#         try:
#             # door
#             if ("indoor" in meta["AUTO_TAG_SCORES"] and "outdoor" in meta["AUTO_TAG_SCORES"]):
#                 if meta["AUTO_TAG_SCORES"]["indoor"] > meta["AUTO_TAG_SCORES"]["outdoor"]:
#                     meta["door"] = "indoor"
#                 else:
#                     meta["door"] = "outdoor"
#             elif "indoor" in meta["AUTO_TAG_SCORES"]:
#                 meta["door"] = "indoor"
#             elif "outdoor" in meta["AUTO_TAG_SCORES"]:
#                 meta["door"] = "outdoor"
#             else:
#                 meta["door"] = ""
#
#             # continent
#             continent_name = get_continent(lat, lon)
#             # continent_name = get_location_of(point, continents_and_countries)
#             meta["continent"] = continent_name
#
#             # camera brand
#             brand = meta["DEVICE"].split("+")[0].lower()
#             meta["camera_brand"] = brand
#
#             new_list.append(meta)
#         except:
#             print("miss "+str(i))
#             pass
#         if i % 1000 == 0:
#             time2 = time.time()
#             print(str(i)+": "+str((time2-time1)/60)+" min")
#             time1 = time2
#
# with open('/mnt/data/home/kxj200023/data/yfcc100m/'+'0-17_processed.json', 'w') as fp:
#     json.dump(new_list, fp)
# print("finish")

# # statistic
#
# camera_brand = {}
# continents = {}
# indoor = 0.0
# outdoor = 0.0
# all = 0.0
# date = []
# date_num = {}
#
# a = 0
# b = 0
# c = 0
#
# with open('/mnt/data/home/kxj200023/data/yfcc100m/'+'0-17_processed.json') as f:
#     list = json.load(f)
#     all = len(list)
#     for i in range(len(list)):
#         meta = list[i]
#         if meta["door"] != "":
#             if meta["door"] == "indoor":
#                 indoor = indoor + 1.0
#             else:
#                 outdoor = outdoor + 1.0
#
#             brand = meta["camera_brand"].lower()
#             if brand not in camera_brand:
#                 camera_brand[brand] = 1.0
#             else:
#                 camera_brand[brand] = camera_brand[brand] + 1.0
#
#             if meta["continent"] not in continents:
#                 continents[meta["continent"]] = 1.0
#             else:
#                 continents[meta["continent"]] = continents[meta["continent"]] + 1.0
#             if meta["DATE_TAKEN"][0:7] not in date:
#                 date.append(meta["DATE_TAKEN"][0:7])
#                 date_num[meta["DATE_TAKEN"][0:7]] = 1
#             else:
#                 date_num[meta["DATE_TAKEN"][0:7]] = date_num[meta["DATE_TAKEN"][0:7]] + 1
#
#             if meta["DATE_TAKEN"][0:3] == "201" and meta["DATE_TAKEN"][0:4] != "2013":
#                 a = a +1
#             elif meta["DATE_TAKEN"][0:3] == "200":
#                 b = b +1
#             elif int(meta["DATE_TAKEN"][0:3]) < 200:
#                 c = c + 1
#
# date.sort()
# for i in sorted(date_num.keys()):
#     print(i, date_num[i])
# print(date)
# print(len(date))
# print("total: ",all)
# print(a)
# print(b)
# print(c)
# print()
# print("door:")
# print("indoor: ", indoor/all)
# print("outdoor: ", outdoor/all)
# print()
# print("continents:")
# for item in continents.keys():
#     print(item+": ",continents[item]/all)
# print()
# print("camera_brands:")
# for item in camera_brand.keys():
#     print(camera_brand[item]/all)
# for item in camera_brand.keys():
#     print(item)


# # split by year
# dict = {}
# year1 = 0
# year2 = 0
# year3 = 0
# list1 = []
# list2 = []
# list3 = []
# sorted_list1 = []
# sorted_list2 = []
# sorted_list3 = []
#
# with open('/mnt/data/home/kxj200023/data/yfcc100m/'+'0-17_processed.json') as f:
#     list = json.load(f)
#     all = len(list)
#     for i in range(len(list)):
#         meta = list[i]
#         if meta["door"] != "":
#             if meta["DATE_TAKEN"][0:3] == "201":
#                 list3.append(i)
#             elif meta["DATE_TAKEN"][0:3] == "200":
#                 list2.append(i)
#             elif meta["DATE_TAKEN"][0:3] < "200":
#                 list1.append(i)
#
#     for i in range(len(list1)):
#         time = "0001-10-08 15:28:38.0"
#         choose = 0
#         for j in range(len(list1)):
#             if list[list1[j]]["DATE_TAKEN"] > time:
#                 time = list[list1[j]]["DATE_TAKEN"]
#                 choose = list1[j]
#         sorted_list1.append(list[choose])
#         list1.remove(choose)
#
#     for i in range(30000):
#         time = "0001-10-08 15:28:38.0"
#         choose = 0
#         for j in range(len(list2)):
#             if list[list2[j]]["DATE_TAKEN"] > time:
#                 time = list[list2[j]]["DATE_TAKEN"]
#                 choose = list2[j]
#         sorted_list2.append(list[choose])
#         list2.remove(choose)
#
#     for i in range(30000):
#         time = "0001-10-08 15:28:38.0"
#         choose = 0
#         for j in range(len(list3)):
#             if list[list3[j]]["DATE_TAKEN"] > time:
#                 time = list[list3[j]]["DATE_TAKEN"]
#                 choose = list3[j]
#         sorted_list3.append(list[choose])
#         list3.remove(choose)
#
# final_list = []
# final_list.append(sorted_list1)
# final_list.append(sorted_list2)
# final_list.append(sorted_list3)
# with open('/mnt/data/home/kxj200023/data/yfcc100m/'+'0-17_threeLists.json', 'w') as fp:
#     json.dump(final_list, fp)


# final preprocessing
dict = {}
with open('/mnt/data/home/kxj200023/data/yfcc100m/'+'0-17_threeLists.json') as f:
    list = json.load(f)
    old_list1 = list[0]
    list2 = list[1]
    list3 = list[2]
    list1 = old_list1
    # list1 = []
    # while len(list1) < 30000:
    #     for i in range(len(old_list1)):
    #         if len(list1) < 30000:
    #             list1.append(old_list1[i])

    count = 0
    for angle in range(70):
        for row in list1:
            if row['door'] == 'indoor':
                door = 0
            else:
                door = 1
            if row['continent'] == 'North America':
                continent = 1.0
            else:
                continent = 0.0
            imgname = str(count)+ '.' + row['EXT']
            dict[imgname] = [0, door, continent]
            with Image.open(row['path']) as im:
                im.rotate(angle).save('/home/kxj200023/data/YFCC/' +
                '0' + '/' + str(door) + '/' + imgname)
            # os.system(
            #     'cp ' + row['path'] + ' /home/kxj200023/data/YFCC/' +
            #     '0' + '/' + str(door) + '/' + imgname)
            count = count + 1

    for row in list2:
        if row['door'] == 'indoor':
            door = 0
        else:
            door = 1
        if row['continent'] == 'North America':
            continent = 1.0
        else:
            continent = 0.0
        imgname = str(count) + '.' + row['EXT']
        dict[imgname] = [1, door, continent]
        os.system(
            'cp ' + row['path'] + ' /home/kxj200023/data/YFCC/' +
            '1' + '/' + str(door) + '/' + imgname)
        count = count + 1

    for row in list3:
        if row['door'] == 'indoor':
            door = 0
        else:
            door = 1
        if row['continent'] == 'North America':
            continent = 1.0
        else:
            continent = 0.0
        imgname = str(count) + '.' + row['EXT']
        dict[imgname] = [2, door, continent]
        os.system(
            'cp ' + row['path'] + ' /home/kxj200023/data/YFCC/' +
            '2' + '/' + str(door) + '/' + imgname)
        count = count + 1

with open('/home/kxj200023/data/YFCC/'+'data.json', 'w') as fp:
    json.dump(dict, fp)











import pandas as pd
import torch

path = "/home/kxj200023/data/NYPD/data.csv"
initial_data = pd.read_csv(path, encoding='latin-1', low_memory=False)

def frame2tensor(initial_data):
    y = initial_data['frisked'].values
    others = initial_data.drop('frisked', axis=1)

    z = others['race_B'].values
    x = others.drop('race_B', axis=1).values

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y)
    z = torch.tensor(z, dtype=torch.float32)
    return x, y, z

datalist = []
datalist.append(initial_data.loc[initial_data['city_BRONX'] == 1])
datalist.append(initial_data.loc[initial_data['city_BROOKLYN'] == 1])
datalist.append(initial_data.loc[initial_data['city_MANHATTAN'] == 1])
datalist.append(initial_data.loc[initial_data['city_QUEENS'] == 1])
datalist.append(initial_data.loc[initial_data['city_STATEN IS'] == 1])

for i in range(len(datalist)):
    data = datalist[i]
    data.drop(labels=['city_BRONX', 'city_BROOKLYN', 'city_MANHATTAN', 'city_QUEENS', 'city_STATEN IS'], axis=1, inplace=True)
    data.to_csv('/home/kxj200023/data/NYPD/' + str(i) + '.csv')


# class MyDataset(Dataset):
#
#     def __init__(self, file_name):
#         path = "/home/kxj200023/data/NYPD/data.csv"
#         initial_data = pd.read_csv(path, encoding='latin-1', low_memory=False)
#
#         y = initial_data['frisked'].values
#         others = initial_data.drop('frisked', axis=1)
#
#         z = others['race_B'].values
#         x = others.drop('race_B', axis=1).values
#
#         self.x = torch.tensor(x, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
#         self.z = torch.tensor(z, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx], self.z[idx]

a = 0