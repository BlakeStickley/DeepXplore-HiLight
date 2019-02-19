FROM python:2.7.12

RUN pip install tensorflow==1.3.0 keras==2.0.8 Pillow h5py opencv-python matplotlib numpy scikit_learn 
RUN pip install --upgrade setuptools

RUN git clone https://github.com/srndic/mimicus.git
RUN cd mimicus && python setup.py develop

RUN mkdir deepxplore-CS239
COPY . /deepxplore-CS239

ENV HOME /deepxplore-CS239
RUN env

WORKDIR $HOME
CMD [ "bash" ]
