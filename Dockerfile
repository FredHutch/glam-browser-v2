FROM python:3.8.2-slim
RUN apt-get update && \
	apt-get install -y hdf5-tools libhdf5-dev libhdf5-serial-dev build-essential && \
	apt-get install -y python3-numpy python3-scipy python3-pandas python3-dev libmariadb-dev
ADD requirements.txt /home/dash/
RUN echo remove mee ## TODO REMOVE ME
RUN pip3 install -r /home/dash/requirements.txt && \
	pip3 install scikit-bio && \
	HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ pip3 install tables
# Note that the binary release version of numcodecs (0.7.1 as of 9/14/20)
# has a bug which causes "illegal instruction set" on the gitlab build machine
# filed issue: https://github.com/zarr-developers/numcodecs/issues/252
# To work around this, we'll install from source.
## RUN pip3 uninstall -y numcodecs
## RUN pip3 uninstall -y numcodecs # make sure
## RUN pip3 install -v --no-cache-dir --no-binary numcodecs numcodecs==0.7.1
RUN useradd -u 5555 -m -d /home/dash -c "dash user" dash
ADD app.py /home/dash/
ADD glam-start.sh /home/dash
ADD helpers/ /home/dash/helpers/
ADD share/ /share/
RUN chown -R dash:dash /home/dash 
WORKDIR /home/dash
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
