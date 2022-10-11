docker build --tag task1 .
#docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task1
docker run -v "/results" task1
