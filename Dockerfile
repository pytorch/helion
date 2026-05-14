FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKSPACE_DIR=/workspace/helion

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libffi-dev \
        libgomp1 \
        ninja-build \
        openssh-server \
        pkg-config \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        rsync && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
    mkdir -p /root/.ssh /var/run/sshd && \
    chmod 700 /root/.ssh && \
    sed -i 's/^#*PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/^#*PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#*PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    printf '\nClientAliveInterval 120\nClientAliveCountMax 3\n' >> /etc/ssh/sshd_config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKSPACE_DIR}

COPY scripts/docker_entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY . ${WORKSPACE_DIR}

RUN mkdir -p /workspace/pytorch

RUN git clone --depth 1 https://github.com/Dao-AILab/quack.git /workspace/quack

RUN python -m pip install \
        --pre \
        --index-url https://download.pytorch.org/whl/nightly/cu130 \
        --extra-index-url https://pypi.org/simple \
        torch \
        triton && \
    python -m pip install -e /workspace/quack && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HELION=0.0+docker \
    python -m pip install \
        -e '.[dev,cute-cu13]' \
        absl-py \
        cffi \
        apache-tvm-ffi \
        jax \
        packaging \
        pytest-xdist \
        pyrefly==0.51.1 \
        ruff==0.15.0 && \
    if [ -d .git ]; then \
        pre-commit install-hooks; \
    else \
        echo "Skipping pre-commit hook bootstrap: no .git in build context"; \
    fi

EXPOSE 22

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["sleep", "infinity"]
