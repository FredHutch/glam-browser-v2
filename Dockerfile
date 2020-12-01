FROM quay.io/aptible/ubuntu:18.04
RUN apt-get update && \
	apt-get install -y software-properties-common && \
	add-apt-repository -y ppa:deadsnakes/ppa && \
	apt-get install -y python3.8 && \
	apt-get install -y hdf5-tools libhdf5-dev libhdf5-serial-dev build-essential && \
	apt-get install -y python3-pip && \
	apt-get install -y python3-numpy python3-scipy python3-pandas python3-dev libmariadb-dev
ADD requirements.txt /home/dash/
RUN pip3 install -r /home/dash/requirements.txt && \
	pip3 install scikit-bio && \
	HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ pip3 install tables
RUN useradd -u 5555 -m -d /home/dash -c "dash user" dash
ADD . /home/dash/
RUN chown -R dash:dash /home/dash 
WORKDIR /home/dash
# Install the CLI
RUN python3 -m pip install .
EXPOSE 8050
ENV DATA_FOLDER=/share
ARG DB_NAME
ENV DB_NAME=$DB_NAME
ARG DB_USERNAME
ENV DB_USERNAME=$DB_USERNAME
ARG DB_PASSWORD
ENV DB_PASSWORD=$DB_PASSWORD
ARG DB_HOST
ENV DB_HOST=$DB_HOST
ARG DB_PORT
ENV DB_PORT=$DB_PORT
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ARG AWS_REGION
ENV AWS_REGION=$AWS_REGION
ARG GTM_CONTAINER
ENV GTM_CONTAINER=$GTM_CONTAINER
CMD gunicorn --timeout 120 --workers 4 --worker-class gevent --bind 0.0.0.0:8050 app:server
