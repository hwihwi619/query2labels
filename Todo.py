import csv
import glob, json
import os
from pathlib import Path

####################################################################################
####################################################################################
####################################################################################
''' Train Image Picker
Description : Meta가 존재하는 데이터의 이름을 불러와, 그 이름에 해당하는 이미지를 원하는 위치로 옮겨주는 메소드
Usage       : 전체 데이터에서 일부만 라벨링된 경우 사용. 

Param       -> image_list : 파일 명만 가지고 있어야 함.(경로명 포함 X)

Considering : option = 'mv' or 'cp'
'''
# class Train_Image_Picker:
#     def __init__(self, image_list:Optional[list], from_path:str, to_path:str) -> None:
#         self.image_list = image_list
#         self.from_path = from_path
#         self.to_path = to_path
    
#     def get_imgList(self, meta_path:str):
#         self.image_list = os.listdir(meta_path)

        
#     def move_img(self):
#         assert self.image_list # self.image_list == None 이면 error
        
#         # for l in self.image_list:
#             # path
            
# if __name__=='__main__':
#     tip = Train_Image_Picker()
#     tip.

####################################################################################
# train imge picker
# meta_path = '/home/hwi/Downloads/VQIS-POC_(BOX) 2022-10-20 9_27_13 /meta/VQIS/images'

# # 하위 모두 가는코드 추가
# image_list = os.listdir(meta_path)
# image_list = [Path(img).stem for img in image_list]
# print(len(image_list))
# print(image_list[1])

# # mv to to path
# from_path = '/home/hwi/github/data/VQIS_PoC/suite'
# to_path = '/home/hwi/Downloads/filetered chick'

# exits_list = os.listdir(from_path)

# for i in image_list:
#     if i in exits_list:
#         from_ = os.path.join(from_path, i)
#         to_ = os.path.join(to_path, i)
#         os.rename(from_, to_)

'''
import os
import shutil

os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
'''
####################################################################################

####################################################################################
# suite2csv , bounding box , MinMax
# class_id center_x center_y width height

# path = '/home/hwi/github/data/VQIS-POC_(BOX) middle chick'

# annotation_path='/home/hwi/github/data/middle_chick_detect/meta'

# origin_width = 1520
# origin_height = 2048

def get_meta_files(path):
    files = glob.glob(path+'/*.json', recursive=True)
    # files = glob.glob(path+'/**/*.json', recursive=True)
    return files

def get_name(file):
    return Path(Path(file).stem).stem         # delete '.json'

# files = get_meta_files()
# # print(len(files))

# for i, file in enumerate(files):
#     # open metafile
#     with open(file, encoding='UTF8') as json_meta_file:
#         json_meta_data = json.load(json_meta_file)
    
#     # get file name
#     name = get_name(file)

#     # open label file
#     label_path = os.path.join(path, json_meta_data['label_path'][0])
#     with open(label_path, encoding='UTF8') as json_label_data:
#         json_label_data = json.load(json_label_data)
#     print(json_label_data)
#     print()
#     print('name', name)
#     print()
    
#     result = ''

#     # get 육계
#     # if label exists on data
#     if 'objects' in json_label_data:
#         # object_num = len(json_label_data['objects'])
#         # for num in range(object_num):
#         #     print(num)
    
#         for obj in json_label_data['objects']:
#             if obj['class_name'] == '육계':
#                 # print('yes')
#                 print(obj['annotation']['coord'])
#                 x = obj['annotation']['coord']['x']
#                 y = obj['annotation']['coord']['y']
#                 width = obj['annotation']['coord']['width']
#                 height = obj['annotation']['coord']['height']
#                 print(x, y, width, height)
                
#                 x_center = (x+(width/2)) / origin_width
#                 y_center = (y+(height/2)) / origin_height
#                 width = width / origin_width
#                 height = height / origin_height

#                 print(x_center, y_center, width, height)
#                 result +=f'0 {x_center} {y_center} {width} {height}\n'
#                 print()
            
#             output_name = os.path.join(annotation_path, name+'.txt')
#             with open(output_name, 'w', encoding='utf-8') as f:
#                 f.write(result)
        
####################################################################################


####################################################################################
####################################################################################
####################################################################################
''' Csv Maker
Description : suite의 raw json파일을 이용하여 csv파일을 만들어줌.
Usage       : suite에서 라벨링 종류에 따라 다르게 만들어서 다른 동작을 같은 함수로 이용할 수 있게 하기

Param       -> 

Considering : 부모 클래스 (형식을 정해주는)
                자식 클래스 : suite 데이터 포멧 종류마다 (segmentation / classification / multi-label classificaiton등등)
'''

# # get files
# train_path = '/home/hwi/Downloads/VQIS-POC label data/meta/VQIS/VQIS_POR_TEST'
# source_path = '/home/hwi/Downloads/VQIS-POC label data'

# files = get_meta_files(train_path)
# # print(files)
# # print(len(files))

# result = []
# # get tag data
# for i, file in enumerate(files):
#     row =[]
#     # open metafile
#     with open(file, encoding='UTF8') as json_meta_file:
#         json_meta_data = json.load(json_meta_file)
    
#     # get file name
#     name = get_name(file)
#     print()
#     print(name)
#     row.append(name)

#     # open label file
#     label_path = os.path.join(source_path, json_meta_data['label_path'][0])
#     with open(label_path, encoding='UTF8') as json_label_data:
#         json_label_data = json.load(json_label_data)
    
#     test_str = ''
#     if 'categories' in json_label_data:
#         opt_num = len(json_label_data['categories']['properties'][0]['option_names'])
#         for num in range(opt_num):
#             index_num = num + 1
#             tag = json_label_data['categories']['properties'][0]['option_names'][num]
#             test_str +=tag
#             row.append(tag)
#     # print('row')
#     # print(row)
#     result.append(row)
#     # if opt_num !=1:
#     #     break

# # print('result')
# # print(result)
# # make csvfile
# csv_file_name = 'VQIS-TEST.csv'

# with open(csv_file_name, 'w') as f:
#     # for r in result:
#     wr = csv.writer(f)
#     wr.writerows(result)
# print('end')

####################################################################################
####################################################################################
####################################################################################
'''
Description : 이미지 파일명을 property를 가진 바꿔버림
Usage       : 

Param       -> 

Considering : os rename을 for문 마지막에 해야할듯. tag에 모든 propery추가하고
'''

# # # get files
# train_path = '/home/hwi/Downloads/VQIS-POC label data/meta/VQIS/VQIS_FOR_TRAIN'
# source_path = '/home/hwi/Downloads/VQIS-POC label data'
# from_path = '/home/hwi/Downloads/VQIS-POC Image data (copy)/VQIS/VQIS_FOR_TRAIN'
# to_path = '/home/hwi/Downloads/VQIS-POC normal/image/train'

# files = get_meta_files(train_path)
# # print(files)
# # print(len(files))

# # get tag data
# for i, file in enumerate(files):
#     # open metafile
#     with open(file, encoding='UTF8') as json_meta_file:
#         json_meta_data = json.load(json_meta_file)
    
#     # get file name
#     name = get_name(file)
#     print()
#     print(name)

#     # open label file
#     label_path = os.path.join(source_path, json_meta_data['label_path'][0])
#     with open(label_path, encoding='UTF8') as json_label_data:
#         json_label_data = json.load(json_label_data)
    
#     if 'categories' in json_label_data:
#         opt_num = len(json_label_data['categories']['properties'][0]['option_names'])
#         for num in range(opt_num):
#             index_num = num + 1
#             tag = json_label_data['categories']['properties'][0]['option_names'][num]
#             # if tag =='정상':
#             #     origin_file = os.path.join(from_path, name+'.jpg')
#             #     new_file ='normal_'+name+'.jpg'
#             #     os.rename(origin_file, os.path.join(to_path, new_file))
#             # if tag =='정상':
#             #     origin_file = os.path.join(from_path, name+'.jpg')
#             #     new_file =name+'_normal'.jpg'
#             #     os.rename(origin_file, os.path.join(to_path, new_file))
#             if tag !='정상':
#                 origin_file = os.path.join(from_path, name+'.jpg')
#                 new_file =name+'.jpg'
#                 os.rename(origin_file, os.path.join(to_path, new_file))
#                 break

####################################################################################
####################################################################################
####################################################################################
'''
Description : 이미지 파일명을 모든 property를 가지도록 바꿔버림
Usage       : 

Param       -> 

Considering : os rename을 for문 마지막에 해야할듯. tag에 모든 propery추가하고
'''

# get files
train_path = '/home/hwi/Downloads/VQIS-POC label data/meta/VQIS/VQIS_FOR_TEST'
source_path = '/home/hwi/Downloads/VQIS-POC label data'
from_path = '/home/hwi/github/VQIS/runs/detect/exp/center_crops/chicken'
to_path = '/home/hwi/Downloads/VQIS-POC dataset-cropped-labeled/image/test'

files = get_meta_files(train_path)
image_list = os.listdir(from_path)
# print(files)
# print(len(files))

result = []
# get tag data
for i, file in enumerate(files):
    row =[]
    # open metafile
    with open(file, encoding='UTF8') as json_meta_file:
        json_meta_data = json.load(json_meta_file)
    
    # get file name
    name = get_name(file)
    print()
    print(name)
    row.append(name)

    # open label file
    label_path = os.path.join(source_path, json_meta_data['label_path'][0])
    with open(label_path, encoding='UTF8') as json_label_data:
        json_label_data = json.load(json_label_data)
    
    test_str = ''
    if 'categories' in json_label_data:
        opt_num = len(json_label_data['categories']['properties'][0]['option_names'])
        for num in range(opt_num):
            index_num = num + 1
            tag = json_label_data['categories']['properties'][0]['option_names'][num]
            test_str +=tag
            row.append(tag)

    str = "*".join(row[1:])
    
    img = name+'.jpg'
    if img in image_list:
        from_ = os.path.join(from_path, img)
        to_ = os.path.join(to_path, str+'*'+img)
        os.rename(from_, to_)
