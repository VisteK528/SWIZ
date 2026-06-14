#!/bin/bash
set -e

source "/opt/ros/$ROS_DISTRO/setup.bash"

if [ -f "/other_ws/install/setup.bash" ]; then
  source "/other_ws/install/setup.bash"
fi

if [ -f "/ws/install/setup.bash" ]; then
  source "/ws/install/setup.bash"
fi

if [ -f "/ws/.venv/bin/activate" ]; then
  source "/ws/.venv/bin/activate"

  PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

  export PYTHONPATH="/ws/.venv/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH"
fi

exec "$@"