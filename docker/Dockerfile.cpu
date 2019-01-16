FROM tensorflow/tensorflow:1.3.0-py3

ADD requirements/requirements-py3.txt /install/requirements-py3.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /install/requirements-py3.txt

# Set up our notebook config.
COPY docker/jupyter_notebook_config.py /root/.jupyter/

WORKDIR /notebooks

CMD /run_jupyter.sh --allow-root
