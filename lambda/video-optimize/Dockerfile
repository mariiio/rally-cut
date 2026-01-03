FROM public.ecr.aws/lambda/python:3.11-arm64

# Install FFmpeg (ARM64 version - 20% cheaper on Graviton2)
RUN yum install -y tar xz && \
    curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz | tar xJ && \
    mv ffmpeg-*-arm64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-arm64-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-*-arm64-static && \
    yum clean all

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copy function code
COPY handler.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "handler.lambda_handler" ]
