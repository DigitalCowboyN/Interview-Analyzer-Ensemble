FROM ghcr.io/ggerganov/llama.cpp:light

#FROM python:3.10

# Set the working directory to your project path
WORKDIR /workspaces/interview_analyzer_ensemble

RUN apt-get update && apt-get install -y \
     ffmpeg \
     sox \
     libsndfile1 \
     git \
     build-essential \
     cmake \
     curl \
     python3.10 \
     python3.10-venv \
     python3.10-dev \
     python3-pip \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
  && ln -sf /usr/bin/pip3 /usr/bin/pip \
  && rm -rf /var/lib/apt/lists/*

# # Clone and build llama.cpp
# # RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
# #     cd /opt/llama.cpp && \
# #     make

RUN pip install --upgrade pip

# # Set llama.cpp as globally available
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/ctransformers/lib/basic:/app:$LD_LIBRARY_PATH"

# Copy the project files
COPY . .

# # Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ctransformers shared library to a standard location (/usr/lib)
# RUN cp /usr/local/lib/python3.10/dist-packages/ctransformers/lib/basic/libctransformers.so /usr/lib/ \
#     && chmod +x /usr/lib/libctransformers.so

# # Command to run when the container starts
CMD ["python", "src/main.py"]