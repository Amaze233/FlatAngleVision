import cv2
import os
from Segmentation.info import *


# 扫描目录及其子目录下文件函数，参数：目录，前缀，后缀
# @author 戴柯
# @version 1.0
def scan_files(directory, prefix=None, postfix=None):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))
    return files_list


path = MASK_PATH
file_list = scan_files(path)
# 以灰度模式打开Mask下所有文件，每个文件遍历所有像素点，灰度大于200视为边界，赋值3，灰度等于0视为背景，赋值2，
# 其他视为前进，赋值1，遍历完成后覆盖保存在Mask文件夹下
# @author 戴柯
# @version 1.1
for file in file_list:
    img = cv2.imread(file, 0)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # if img[x, y] == 0:
            #     img[x, y] = 2
            # elif img[x, y] > 200:
            #     img[x, y] = 3
            # else:
            #     img[x, y] = 1
            if 0 < img[x, y] < 200:
                img[x, y] = 1
            else:
                img[x, y] = 0
    _, name = os.path.split(file)
    cv2.imwrite(path+'\\'+name, img)
