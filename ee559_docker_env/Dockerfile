FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-lc"]

# Use a conda-managed base environment so FAISS and notebook tooling work reliably.
RUN conda install -n base -c conda-forge -y python=3.10 pip faiss-cpu jupyter nbconvert && \
	conda clean -afy

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
