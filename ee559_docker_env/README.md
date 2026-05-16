# Docker image for RAG notebook

This image is meant to run `RAG/rag_step3_index.ipynb` on the RCP cluster with FAISS available.
Build and push it from a machine that has Docker installed and access to `registry.rcp.epfl.ch`.

## Build

From the repository root:

```bash
docker build -t registry.rcp.epfl.ch/ee559/environment-with-faiss:latest -f ee559_docker_env/Dockerfile ee559_docker_env
```

## Push

```bash
docker push registry.rcp.epfl.ch/ee559/environment-with-faiss:latest
```

## Run on RCP

Submit the job with the new image:

```bash
runai submit --name rag-step3-index-g44-faiss --run-as-uid 258393 --image registry.rcp.epfl.ch/ee559/environment-with-faiss:latest --gpu 1 --existing-pvc claimname=course-ee-559-scratch-g44,path=/scratch --existing-pvc claimname=home,path=/home/potocnik --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw --command -- bash -lc "cd /scratch/deep_learning && jupyter nbconvert --to notebook --execute RAG/rag_step3_index.ipynb --output-dir results --output rag_step3_index.executed"
```

## Notes

- The image uses a conda base so `faiss-cpu` works without `python3-venv` or system package changes.
- If you want a GPU FAISS build, replace `faiss-cpu` with `faiss-gpu` in the Dockerfile and confirm CUDA compatibility first.
