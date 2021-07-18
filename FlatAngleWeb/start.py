'''
@Project ：FlatAngleWeb
@File    ：start.py
@Author  ：谢逸帆
@Date    ：2021/7/14 9:23
'''
import os.path
import time
import uuid
import requests
from flask import Flask, request, abort, jsonify, render_template, url_for, make_response
import json

from werkzeug.utils import secure_filename

from Segmentation import utils
import cv2



app = Flask(__name__)

# 服务器保存结果图片的文件夹名
CACHE_FOLDER = "image_cache"

# 服务器保存原图的文件夹名
ORIGINAL_IMAGE_DIC = "./Segmentation/bef/"


@app.route('/segmentation/online', methods=['POST', 'GET'])
def segmentation_online():
    """
    请求在线图片分割的接口

    :return: 目前暂时只返回mask图像和预测时间
    """
    try:
        # 解析POST请求
        image_name = decode_POST('online_img_url')

        # 调用模型进行预测并(处理原图)
        start_time = time.time()

        file_ds = utils.load_dataset(image_name)
        image_list = [utils.process(file, ORIGINAL_IMAGE_DIC + image_name) for file in file_ds]

        end_time = time.time()
        cost_time = end_time - start_time

        result_path = ''
        for mask in image_list:
            result_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
            cv2.imwrite(result_path, mask)

        # 服务器本地保存结果
        return jsonify(generate_result(result_path, cost_time))
    except:

        abort(500)


@app.route('/segmentation/upload_file', methods=['POST', 'GET'])
def segmentation_local():
    """
    请求图片文件分割的接口
    :return: 目前暂时只返回mask图像和预测时间
    """
    try:
        print('1')
        image_name = decode_POST('image_file')

        # 调用模型进行预测并(处理原图)
        start_time = time.time()

        file_ds = utils.load_dataset(image_name)
        image_list = [utils.process(file, ORIGINAL_IMAGE_DIC + image_name) for file in file_ds]

        end_time = time.time()
        cost_time = end_time - start_time

        result_path = ''
        for mask in image_list:
            result_path = CACHE_FOLDER + "/" + str(uuid.uuid1()) + ".png"
            cv2.imwrite(result_path, mask)

        # 服务器本地保存结果
        return jsonify(generate_result(result_path, cost_time))
    except:
        abort(500)


def decode_POST(data_name):
    """
    解析POST请求数据并将图片保存至本地

    :param data_name:POST请求中的数据名
    :return: 保存后的图片名
    """
    if data_name == 'online_img_url':
        data = request.get_data()
        json_data = json.loads(data.decode("utf-8"))
        image_url = json_data["online_img_url"]
        print(image_url)
        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            img_name = str(uuid.uuid1()) + '.png'
            with open(ORIGINAL_IMAGE_DIC + img_name, 'wb') as f:
                f.write(r.content)
            return img_name

    elif data_name == 'image_file':

        data = request.files.get('image_file')

        if data is None:
            abort(500)
        else:
            img_name = str(uuid.uuid1()) + '.png'
            print(img_name)
            data.save(ORIGINAL_IMAGE_DIC + img_name)
            return img_name


def generate_result(result_path, cost_time):
    """
    生成返回结果（还需要添加其他参数）

    :param result_path: 结果文件路径
    :param cost_time: 预测所花费时间
    :return: 字典格式的结果
    """
    res = {"mask": "/" + result_path, "cost_time": cost_time}
    return res


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'nmsl'}), 500)


@app.route('/FlatAngle')
def FlatAngle():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def uploadTest():
    download_path = ""
    imageUrl = ""
    if request.method == 'POST':
        imageUrl = request.form.get('online_img_url')
        file = request.files.get('')

        if imageUrl != "":
            onlineUrl = "http://121.199.2.56:5000/segmentation/online"
            data = {'online_img_url': imageUrl}
            print(imageUrl)
            # print(str(imageUrl))
            result1 = requests.post(onlineUrl, data=json.dumps(data))

            print(result1.text)
            json_dict1 = json.loads(result1.text)

            print(json_dict1["result"])

            download_path = os.path.join(json_dict1["result"])

        else:
            uploadUrl = "http://121.199.2.56:5000/segmentation/upload_file"
            # file = request.files.get("image_file")
            # print(type)
            # basePath = os.path.dirname(__file__)  # 当前文件所在路径
            #
            # upload_path = os.path.join(basePath, 'static/images',secure_filename(file.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
            # # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
            # file.save(upload_path)
            #
            # # 使用Opencv转换一下图片格式和名称
            # img = cv2.imread(upload_path)
            # cv2.imwrite(os.path.join(basePath, 'static/images', 'input.png'), img)

            result2 = requests.post(uploadUrl)
            print(result2.text)
            json_dict2 = json.loads(result2.text)
            print(json_dict2["result"])
            download_path = os.path.join(json_dict2["result"])

        # return render_template('segmentation_upload.html', path = download_path, url = imageUrl)

    return render_template('segmentation_upload.html', path=download_path, url=imageUrl)


if __name__ == '__main__':
    app.run()
