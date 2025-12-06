FROM python:3.12

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy your project files
COPY app.py .
COPY MLModel.py .
COPY constants.py .
COPY streamlit_app.py .
COPY mlruns ./mlruns
COPY mlartifacts ./mlartifacts

# Set environment variables
ENV MLFLOW_TRACKING_URI ="file:///app/mlruns"

# Command to run your app
CMD ["bash", "-c", "mlflow server --host 0.0.0.0 --port 5102 --backend-store-uri file:/app/mlruns --default-artifact-root /app/mlartifacts --serve-artifacts & python app.py"]

#docker build -t pipelines_with_mlflow_docker .    

#conda env export --name környezet_neve > environment.yml

#docker build -t mlflow_docker_image .

#docker run --name self_container_name -p 5102:5102 -p 8080:8080 mlflow_docker_image

#docker exec -it my_cont2 /bin/bash