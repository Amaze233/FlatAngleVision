"""
@Project ：FlatAngleWeb
@File    ：test.py
@Author  ：谢逸帆
@Date    ：2021/7/14 15:03
"""

import requests
from Segmentation import utils
import uuid
import cv2

CACHE_FOLDER = "image_cache"

image_url = r"https://inews.gtimg.com/newsapp_bt/0/7396554241/1000"
r = requests.get(image_url, stream=True)
print(r.status_code)
if r.status_code == 200:
    img_name = str(uuid.uuid1()) + '.png'
    img_path = "./Segmentation/bef/" + img_name
    with open(img_path, 'wb') as f:
        f.write(r.content)
    file_ds = utils.load_dataset(img_name)
    result_list = [utils.process(file, img_path) for file in file_ds]

    for mask in result_list:
        mask_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
        cv2.imwrite(mask_path, mask)
