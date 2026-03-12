FROM ubuntu:24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    curl git unzip wget jq sudo \
    python3 python3-pip python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user matching host UID
ARG USER_UID=1000
ARG USER_GID=1000
RUN usermod -l dev -d /home/dev -m ubuntu && \
    groupmod -n dev ubuntu && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Bun (as dev user)
USER dev
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/home/dev/.bun/bin:$PATH"

# Node (needs root)
USER root
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (auto-detect architecture)
RUN ARCH=$(uname -m) \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp/ \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Working directory
WORKDIR /workspace
USER dev
