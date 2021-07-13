import os
import tensorflow as tf
import create_dataset
import pycocotools
from pycocotools.coco import COCO
import cv2

VAL_IMAGES_PATH = "./COCO2017/val/val2017"
VAL_ANNOTATION_PATH = "./COCO2017/val/annotations/instances_val2017.json"


def filename_to_id(filename_list):
    """
    返回coco_image所对应的id列表
    :param filename_list: coco数据集的文件列表
    :return: 文件列表所对应的id列表
    """
    image_id = []
    for i in range(len(filename_list)):
        for j in range(12):
            if filename_list[i][j] != '0':
                image_id.append(int(filename_list[i][j:12]))  # 将字符串转换成数字存储
                break
    return image_id


def clear_unused_images(images_path, annotation_path):
    coco = pycocotools.coco.COCO(annotation_path)
    all_images_path = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    all_images_ids = filename_to_id(os.listdir(images_path))
    for index in range(len(all_images_ids)):
        if len(coco.getAnnIds(all_images_ids[index])) == 0:
            os.remove(all_images_path[index])


def get_biggest_mask(coco_image_id, coco):
    """
    返回图像对应物品集中，面积最大的物体的mask，用于主体分割
    :param coco_image_id: id of the image
    :param coco: the coco operator object
    :return: the mask of the biggest object in the image
    """
    ann_ids = coco.getAnnIds(coco_image_id)
    anns = coco.loadAnns(ann_ids)
    max_area_ann = 0
    for index in range(len(anns)):
        if anns[index]['area'] > anns[max_area_ann]['area']:
            max_area_ann = index
    return coco.annToMask(anns[max_area_ann])


def load_coco_data_detection(images_path, annotation_path):
    """
    返回对应切割后的图像和对应的label集合,用于目标检测
    :param images_path: path to the coco image set
    :param annotation_path: path to the annotation to load
    :return:images and corresponding labels
    """
    # 载入coco数据集处理对象
    coco = pycocotools.coco.COCO(annotation_path)
    # 删除没有对应ann的图片
    clear_unused_images(images_path, annotation_path)
    # 载入图像路径list
    all_images_path = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    # 生成对应的图像id
    images = []
    labels = []
    index = 0
    for path in all_images_path:
        num = 0
        print("path:", index)
        image_id = filename_to_id(os.listdir(images_path))[index]
        ann_id = coco.getAnnIds(image_id)
        anns = coco.loadAnns(ann_id)
        for ann in anns:
            num += 1
            print("image:", num)
            image = cv2.imread(path)
            mask = coco.annToMask(ann)
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    if mask[x][y] == 0:
                        image[x][y] = 0
            cv2.imshow("window", image)
            label = coco.loadCats(ann['category_id'])[0]['name']
            print(label)
            cv2.waitKey(0)
            images.append(image)
            labels.append(label)
        index += 1
    return images, labels


def load_coco_data_segmentation(images_path, annotation_path):
    """
    返回图片的路径集合和相应的二值mask
    :param images_path: path to the coco image set
    :param annotation_path: path to the annotation to load
    :return: images' path and corresponding mask
    """
    # 载入相关数据
    coco = pycocotools.coco.COCO(annotation_path)
    all_images_path = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    all_images_ids = filename_to_id(os.listdir(images_path))
    # 删除没有对应ann的图片
    clear_unused_images(images_path, annotation_path)
    # 获得最大Mask对象
    masks = []
    for img_id in all_images_ids:
        mask = get_biggest_mask(img_id, coco)
        masks.append(mask)
    return all_images_path, masks
