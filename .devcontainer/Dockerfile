# .devcontainer/Dockerfile
# Start from NVIDIA's PyTorch image (includes CUDA + cuDNN + PyTorch)
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set working directory inside container
WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y \
    git curl vim build-essential sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Create a non-root user (UID 1000 matches typical host user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up a pretty, colored prompt for the new user
RUN echo 'export PS1="[\[\033[0;36m\]\u@\h \[\033[0;33m\]\D{%Y-%m-%d %H:%M:%S}\[\033[0m\]] \[\033[0;32m\]\w\[\033[0m\]\\$ "' >> /home/$USERNAME/.bashrc \
    && chown $USER_UID:$USER_GID /home/$USERNAME/.bashrc

# Switch to the non-root user
USER $USERNAME
