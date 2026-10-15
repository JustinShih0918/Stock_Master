FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG DEV_UID=1000
ARG DEV_GID=1000

ENV TZ=Asia/Taipei \
    NPM_CONFIG_PREFIX=/home/stock/.npm-global \
    PATH=/home/stock/.npm-global/bin:/home/stock/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash-completion build-essential ca-certificates curl git jq less tmux \
      netcat-openbsd nodejs npm openssh-client procps python3 python3-pip \
      python3-venv postgresql-client tzdata vim-tiny \
    && rm -rf /var/lib/apt/lists/* \
    && if ! getent group stock >/dev/null; then \
         if getent group "${DEV_GID}" >/dev/null; then groupadd stock; \
         else groupadd --gid "${DEV_GID}" stock; fi; \
       fi \
    && if ! getent passwd stock >/dev/null; then \
         existing_user="$(getent passwd "${DEV_UID}" | cut -d: -f1)"; \
         if [ -n "${existing_user}" ]; then \
           usermod --login stock --home /home/stock --move-home --shell /bin/bash "${existing_user}"; \
           usermod --gid stock stock; \
         else \
           useradd --uid "${DEV_UID}" --gid stock --create-home --shell /bin/bash stock; \
         fi; \
       fi \
    && mkdir -p /workspace /home/stock/.npm-global /home/stock/.claude \
    && chown -R stock:stock /workspace /home/stock \
    && chmod 0777 /workspace \
    && echo 'umask 0000' >> /etc/bash.bashrc \
    && git config --system --add safe.directory '*'

# umask 0000 for every container process so bind-mounted files stay writable by all users.
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN npm --prefix /usr/local install --global n \
    && /usr/local/bin/n 22 \
    && node --version

USER stock
WORKDIR /workspace

RUN npm install --global @anthropic-ai/claude-code

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
