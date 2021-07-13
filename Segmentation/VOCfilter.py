import pathlib
import os

seg_files_root = pathlib.Path("./VOC2012/SegmentationObject")
jpeg_files_root = pathlib.Path("./VOC2012/JPEGImages")


jpeg_files = list(jpeg_files_root.glob('*.jpg'))
seg_files = list(seg_files_root.glob('*.png'))

num = 0
for file in jpeg_files:
    flag = 0
    for seg_file in seg_files:
        if file.stem == seg_file.stem:
            flag = 1
            num += 1
            print(num)
    if flag == 0:
        os.remove(file)