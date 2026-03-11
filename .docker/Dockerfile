ARG ROS_DISTRO="jazzy"
FROM ros:$ROS_DISTRO

SHELL ["/bin/bash", "-lc"]

RUN apt-get update \
    && apt-get upgrade -y \
    && \
    : "remove cache" && \
    apt-get autoremove -y -qq && \
    apt install libserial-dev -y && \
    apt install ros-$ROS_DISTRO-ros2-control ros-$ROS_DISTRO-ros2-controllers ros-$ROS_DISTRO-gz-ros2-control -y && \
    rm -rf /var/lib/apt/lists/*

# Install Gazebo
RUN apt-get update && apt-get install curl lsb-release gnupg libgl1 libglx0 mesa-utils -y && rm -rf /var/lib/apt/lists/*
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update \
    && apt-get install gz-harmonic -y \
    && rm -rf /var/lib/apt/lists/*

# Install ROS dependencies
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
    ros-$ROS_DISTRO-spatio-temporal-voxel-layer \
    ros-$ROS_DISTRO-rqt-graph \
    ros-$ROS_DISTRO-perception-pcl \
    ros-$ROS_DISTRO-pcl-msgs \
    ros-$ROS_DISTRO-navigation2 \
    ros-$ROS_DISTRO-nav2-bringup \
    ros-$ROS_DISTRO-nav2-minimal-tb* \
    ros-$ROS_DISTRO-robot-localization \
    ros-$ROS_DISTRO-vision-opencv \
    ros-$ROS_DISTRO-rtabmap-ros \
    ros-$ROS_DISTRO-plotjuggler-ros \
    ros-$ROS_DISTRO-rqt \
    ros-$ROS_DISTRO-rqt-common-plugins \
    ros-$ROS_DISTRO-rviz2 \
    ros-$ROS_DISTRO-joy \
    libxcb-cursor0 \
    libxcb-cursor-dev \
    && rm -rf /var/lib/apt/lists/*

# Install other some dependencies
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libtbb-dev \
    gdb \
    nano \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Formatting tools
RUN apt-get update && apt-get install -y \
    clang-format \
    && pip install ruff --break-system packages \
    && rm -rf /var/lib/apt/lists/*

# Colors in terminal
RUN echo 'source /usr/lib/git-core/git-sh-prompt 2>/dev/null' >> ~/.bashrc && \
    echo "PROMPT_COMMAND='PS1_CMD1=\$(__git_ps1 \" (%s)\")'" >> ~/.bashrc && \
    echo 'export PS1="\[\033[01;32m\]\u@\H\[\033[00m\]:\[\033[01;34m\]\w\[\033[01;31m\]\${PS1_CMD1}\[\033[00m\]$ "' >> ~/.bashrc

# Shell setup
RUN echo "source /opt/ros/\$ROS_DISTRO/setup.bash" >> ~/.bashrc && \
    echo "if [ -f /ws/install/setup.bash ]; then source /ws/install/setup.bash; fi" >> ~/.bashrc

WORKDIR /ws

COPY .docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]