FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a /data directory in the image for DB storage.
# On HF Spaces free tier this is in-container storage (persists within a session).
# To persist across restarts, upgrade and attach an HF Storage Bucket mounted at /data.
RUN mkdir -p /data && chmod 777 /data

RUN useradd -m -u 1000 user
RUN chown -R user:user /code
USER user

COPY --chown=user:user . /code

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
