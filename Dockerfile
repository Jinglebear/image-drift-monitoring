FROM python:3.8

# update
RUN apt-get update

# import code & data
RUN mkdir /app
WORKDIR /app

ADD src /app/src
ADD data /app/data




# install dependencies
RUN pip install -r src/modules/whylogs/requirements_whylogs_torchdrift.txt
CMD ["bash"]