#!/usr/bin/env bash
# get the image and container name 
PATH_IMAGE_NAME="${PWD}/.container_image_name"
if [ -f "${PATH_IMAGE_NAME}" ]
then
  IMAGE_NAME=$(cat "${PATH_IMAGE_NAME}")
  CONTAINER_NAME=$(sed 's/\//-/g' "${PATH_IMAGE_NAME}") #replace "/" with "-" as container names cannot contain "/"
else
  echo "please build your dev container image first..."
  echo "try running: bash build.sh"
  exit
fi

IMAGE_HASH="latest"


## useful reminders for users 
LIGHT_BLUE='\033[1;34m' 
echo -e "${LIGHT_BLUE}*hint: use command 'bash start_container.sh' to enter this same active container in another terminal tab"
echo -e "${LIGHT_BLUE}*hint: run 'python -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser' and ctrl + click the last URL to start and open jupyter notebook in your browser - or copy pasta the url '0.0.0.0:8888' to your browser"
echo -e "${LIGHT_BLUE}*hint: use 'python -m <python module>' to run python modules e.g. 'python -m pytest'"
echo -e "${LIGHT_BLUE}*hint: use 'docker ps' to list active containers "
echo -e "${LIGHT_BLUE}*hint: use 'docker kill <container name>' to kill / turn off an active container"
echo -e "${LIGHT_BLUE}*hint: tab is your friend =)"
echo " "


# check if continer exisits 
if [ $( docker container ls -a | grep "${CONTAINER_NAME}" | wc -l ) -gt 0 ] 
then
  # check if container is running 
  if [ "$( docker container inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" )" == "true" ] 
  then
  echo "Entering running container: ${CONTAINER_NAME}"
  docker exec -it "${CONTAINER_NAME}" zsh
  else 
  echo "Starting your existing container with:"
  docker start ${CONTAINER_NAME}
  docker attach ${CONTAINER_NAME}
  fi
else
  echo "Creating container: ${CONTAINER_NAME}"
  docker run -it \
  --name "${CONTAINER_NAME}" \
  -v "${PWD}":/home/${USER}/dev_dir \
  -v ~/.aws:/home/${USER}/.aws \
  -v ~/.ssh:/home/${USER}/.ssh \
  -v ~/.gitconfig:/home/${USER}/.gitconfig \
  -p 8888:8888 \
  -p 5000:5000 \
  -p 8080:8080 \
  "${IMAGE_NAME}":${IMAGE_HASH}
fi




