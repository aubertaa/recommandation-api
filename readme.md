# Python FastPI Recommendation API

This is a simple recommendation API built with FastAPI and Python. It uses a simple collaborative filtering algorithm to recommend games to users based on their previous purchases.

## Build

To build the project, you need to have Docker installed on your machine. Then, you can run the following command:

```bash
docker build -t fastapi-recommendation .
```

## Run

To run the project, you can use the following command:

```bash
docker run -d -p 8000:8000 fastapi-recommendation
```

The API will be available at `http://localhost:8000`.