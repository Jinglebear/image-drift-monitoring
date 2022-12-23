FROM python:3.8

# update
RUN apt-get update
# create working directory
RUN mkdir /app
WORKDIR /app

# import code
ADD src /app/src
# import data
# COPY data.tar.gz /app/



# install dependencies
RUN pip install -r src/modules/whylogs/requirements_whylogs_torchdrift.txt

# docker run <image> /bin/bash -c "tar -xzf data.tar.gz;python src/modules/whylogs/main.py"