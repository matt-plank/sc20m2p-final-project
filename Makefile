base.build:
	# Build the docker image for training models
	docker image build \
		-t transfer \
		-f transferEngine/Dockerfile \
		transferEngine

train.build:
	# Build the docker image for training models
	docker image build \
		-t transfer-train \
		-f transferEngine/Dockerfile.train \
		transferEngine

train.run:
	# Run the training script
	docker run \
		--gpus all \
		--volume $(shell pwd)/transferEngine/trainingImages:/transferEngine/trainingImages \
		--volume $(shell pwd)/transferEngine/model.tf:/transferEngine/model.tf \
		transfer-train

train.all:
	# Build the docker image and run the training script
	make base.build
	make train.build
	make train.run

backend.build:
	# Build the docker image for the backend
	docker image build \
		-t transfer-backend \
		-f transferEngine/Dockerfile.backend \
		transferEngine

backend.run:
	# Run the backend
	docker run \
		--gpus all \
		-p 5000:5000 \
		transfer-backend

backend.all:
	# Build the docker image and run the backend
	make base.build
	make backend.build
	make backend.run

frontend.build:
	# Build the docker image for the frontend
	docker image build \
		-t transfer-frontend \
		frontend

frontend.run:
	# Run the frontend
	docker run \
		-p 3000:3000 \
		transfer-frontend

frontend.all:
	# Build the docker image and run the frontend
	make frontend.build
	make frontend.run
