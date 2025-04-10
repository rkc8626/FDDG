import json


import os


import csv



DEST = '/home/kxj200023/data/FairFace/'

dict = {}
races = {}
races['Black'] = 0
races['East Asian'] = 1
races['Indian'] = 2
races['Latino_Hispanic'] = 3
races['Middle Eastern'] = 4
races['Southeast Asian'] = 5
races['White'] = 6

count = 0

with open('/home/kxj200023/data/FairFace_Original/fairface_label_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row[1] == '50-59' or row[1] == '60-69' or row[1] == 'more than 70':
                age = 1
            else:
                age = 0
            if row[2] == 'Male':
                gender = 0.0
            else:
                gender = 1.0
            race = races[row[3]]
            imgname = str(count)+'.jpg'
            dict[imgname] = [race, age, gender]
            os.system(
                'cp ' + '/home/kxj200023/data/FairFace_Original/' + row[0] + ' /home/kxj200023/data/FairFace/' +
                str(race) + '/' + str(age) + '/' + imgname)
            count += 1

with open('/home/kxj200023/data/FairFace_Original/fairface_label_val.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row[1] == '50-59' or row[1] == '60-69' or row[1] == 'more than 70':
                age = 1
            else:
                age = 0
            if row[2] == 'Male':
                gender = 0.0
            else:
                gender = 1.0
            race = races[row[3]]
            imgname = str(count)+'.jpg'
            dict[imgname] = [race, age, gender]
            os.system(
                'cp ' + '/home/kxj200023/data/FairFace_Original/' + row[0] + ' /home/kxj200023/data/FairFace/' +
                str(race) + '/' + str(age) + '/' + imgname)
            count += 1

with open(DEST+'data.json', 'w') as fp:
    json.dump(dict, fp)