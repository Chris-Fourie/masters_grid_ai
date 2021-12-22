#!/usr/bin/env bash

## image name ##
CONTAINER_IMAGE_PATH="${PWD}/.container_image_name"
echo ${CONTAINER_IMAGE_PATH}
if [ -f "${CONTAINER_IMAGE_PATH}" ]
then
  CONTAINER_IMAGE_NAME=$(cat "${CONTAINER_IMAGE_PATH}")
else 
  CONTAINER_IMAGE_NAME=chris_fourie/${PWD##*/} 
  echo image name: "${CONTAINER_IMAGE_NAME}"
  echo "${CONTAINER_IMAGE_NAME}" >> "${CONTAINER_IMAGE_PATH}"
fi

docker build -t ${CONTAINER_IMAGE_NAME} \
--build-arg USER=${USER} --build-arg USERID=$(id -u) \
.