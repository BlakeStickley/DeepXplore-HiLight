FROM python:2.7.12

RUN pip install tensorflow==1.3.0 keras==2.0.8 Pillow h5py opencv-python matplotlib numpy scikit_learn && pip install --upgrade setuptools
RUN git clone https://github.com/srndic/mimicus.git && cd mimicus && python setup.py develop

COPY . /deepxplore-CS239

ENV HOME /deepxplore-CS239
RUN env

WORKDIR $HOME
CMD [ "bash" ]
