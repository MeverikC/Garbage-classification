# app.py
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

app = Flask(__name__)


# 配置参数
class WebConfig:
    model_path = "./garbage_classifier_best.pth"
    label_mapping = "./class_indices.json"
    upload_folder = 'uploads'
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224  # 需要与训练时保持一致


# 确保上传目录存在
os.makedirs(WebConfig.upload_folder, exist_ok=True)

# 数据预处理（与验证集相同）
transform = transforms.Compose([
    transforms.Resize(int(WebConfig.img_size * 1.1)),
    transforms.CenterCrop(WebConfig.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 加载模型
def load_model():
    # 加载类别映射
    with open(WebConfig.label_mapping, 'r') as f:
        label_data = json.load(f)
        class_names = label_data['classes']
        class_to_idx = label_data['class_to_idx']

    # 创建模型（需要与训练时的结构一致）
    model = models.efficientnet_b3(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(class_names))
    )

    # 加载权重
    checkpoint = torch.load(WebConfig.model_path, map_location=WebConfig.device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(WebConfig.device)
    model.eval()

    return model, class_names


model, class_names = load_model()


# 辅助函数
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in WebConfig.allowed_extensions


def predict_image(image_path):
    try:
        # 打开并预处理图像
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(WebConfig.device)

        # 推理
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # 获取top3结果
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        results = []
        for i in range(3):
            results.append({
                "class": class_names[top3_indices[i].item()],
                "confidence": f"{top3_probs[i].item() * 100:.2f}%"
            })

        return {"success": True, "predictions": results}

    except Exception as e:
        return {"success": False, "error": str(e)}


# 路由定义
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # 保存上传的文件
            filepath = os.path.join(WebConfig.upload_folder, file.filename)
            file.save(filepath)

            # 进行预测
            result = predict_image(filepath)

            # 删除临时文件
            if os.path.exists(filepath):
                os.remove(filepath)

            if result['success']:
                return render_template('result.html',
                                       predictions=result['predictions'],
                                       filename=file.filename)
            else:
                return jsonify(result), 500

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
