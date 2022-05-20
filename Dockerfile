FROM nvcr.io/nvidia/pytorch:21.11-py3

ARG USER_ID
ARG GROUP_ID

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user