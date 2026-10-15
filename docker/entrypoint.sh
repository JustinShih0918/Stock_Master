#!/bin/sh
# Files created in the bind-mounted workspace must stay writable by every host user,
# whatever UID the container runs as, so the checkout can move between machines as-is.
umask 0000
exec "$@"
