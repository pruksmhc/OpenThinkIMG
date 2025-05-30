# === 基础镜像 ===
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# === 复制项目代码到容器中 ===
# 请确保构建命令的上下文路径（build context）是 tool-agent 的上一层目录
COPY ../../ /app/tool_agent/

# === 拷贝模型权重 ===
# 请注意：Docker 要求构建上下文里的文件才能 COPY
# 所以建议你将权重提前链接到 build context，或在运行时挂载
COPY weights/Molmo-7B-D-0924 /weights/Molmo-7B-D-0924
COPY weights/sam2-hiera-large /weights/sam2-hiera-large
COPY weights/groundingdino_swint_ogc.pth /weights/groundingdino_swint_ogc.pth
COPY weights/GroundingDINO_SwinT_OGC.py /weights/GroundingDINO_SwinT_OGC.py

# === 创建日志目录 ===
RUN mkdir -p /log
RUN apt update
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y git

RUN git config --global http.sslVerify false

RUN pip install git+https://github.com/facebookresearch/sam2.git
RUN pip install git+https://github.com/IDEA-Research/GroundingDINO.git
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118


RUN python -c "import easyocr; easyocr.Reader(['ch_sim','en'])"

# === 设置环境变量 ===
ENV PYTHONUNBUFFERED=1

# === 设置工作目录 ===
WORKDIR /app/tool_agent

# === 设置默认启动命令（等价于 %runscript）===
CMD ["python", "tool_server/tool_workers/scripts/launch_scripts/start_server_local.py", \
     "--config",  \
     "tool_server/tool_workers/scripts/launch_scripts/config/service_apptainer.yaml"]

# === 镜像元信息（可选）===
LABEL maintainer="Mingyang Song <mysong23@m.fudan.edu.cn>"
LABEL version="1.0.0"
LABEL description="Tool Server Image"