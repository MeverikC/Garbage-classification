FROM python:3.12-alpine

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

RUN python.exe -m pip install --upgrade pip

# 安装项目依赖
RUN pip install -r unix-requirements.txt

# 下载预训练模型（替换为实际发布 URL）
RUN wget -O /app/garbage_classifier_best.pth \
    https://github.com/MeverikC/Garbage-classification/releases/download/v2.0/garbage_classifier_best_EfficientNet-B4.pth

# 暴露端口
EXPOSE 9005

# 使用 Gunicorn 启动服务
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:9005", "--log-level", "debug", "app:app"]