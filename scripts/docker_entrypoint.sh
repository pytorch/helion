#!/usr/bin/env bash
set -euo pipefail

add_keys() {
    local key_blob="$1"
    [[ -n "${key_blob}" ]] || return 0

    touch /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys

    while IFS= read -r key; do
        [[ -n "${key}" ]] || continue
        grep -qxF "${key}" /root/.ssh/authorized_keys || echo "${key}" >> /root/.ssh/authorized_keys
    done <<< "${key_blob}"
}

mkdir -p /root/.ssh /var/run/sshd
chmod 700 /root/.ssh

public_keys="${PUBLIC_KEY:-}"
ssh_public_keys="${SSH_PUBLIC_KEY:-}"

if [[ -n "${public_keys}" || -n "${ssh_public_keys}" ]]; then
    add_keys "${public_keys}"
    add_keys "${ssh_public_keys}"
    ssh-keygen -A
    /usr/sbin/sshd
fi

exec "$@"
