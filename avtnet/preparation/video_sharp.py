import os
import cv2
import concurrent.futures
from tqdm import tqdm


def ums_sharpen(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # 获取视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用UMS方法对帧进行锐化
        sharpened = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)

        # 写入锐化后的帧
        out.write(sharpened)

    # 释放资源
    cap.release()
    out.release()


def process_video(video_path):
    # 检查文件扩展名是否为.mp4
    if not video_path.lower().endswith(".mp4"):
        return

    # 构造输出路径
    output_dir = os.path.dirname(video_path).replace("video", "video_sharp")
    os.makedirs(output_dir, exist_ok=True)
    output_path = video_path.replace("video", "video_sharp")

    # 执行UMS锐化
    ums_sharpen(video_path, output_path)


if __name__ == "__main__":
    input_dir = "/workspace/AVTSR/data/lrs3/video"
    video_paths = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files]

    # 分配任务给线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for video_path in video_paths:
            future = executor.submit(process_video, video_path)
            futures.append(future)

        # 显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print("视频锐化完成！")