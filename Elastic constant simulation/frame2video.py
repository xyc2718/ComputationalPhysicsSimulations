import cv2
import os

def images_to_video(image_folder, video_file, frame_rate):
    # 获取所有以frame_开头的图片文件，按数字顺序排序
    images = [img for img in os.listdir(image_folder) if img.startswith('frame') and img.endswith('.jpg')]
    images.sort(key=lambda x: int(x.replace('frame', '').split('.')[0]))


    # 读取第一张图片以获取帧的宽度和高度
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape

    # 定义视频编码器，创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以根据需求更改编码器
    video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))

    # 逐帧写入视频
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # 释放VideoWriter对象
    video.release()

# 示例用法
image_folder = 'output\\AdHvRk3_Al2_222_100K_celltemp0\\Frames'
parent_folder = os.path.dirname(image_folder)  # 获取 image_folder 的上级目录
video_file = os.path.join(parent_folder, "video.mp4")  # 将 video.mp4 放在上级目录中
frame_rate = 30  # 帧率，可以根据需求更改

images_to_video(image_folder, video_file, frame_rate)
