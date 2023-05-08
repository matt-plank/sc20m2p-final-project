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

train:
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
		-it \
		-p 5000:5000 \
		transfer-backend

backend:
	# Build the docker image and run the backend
	make base.build
	make backend.build
	make backend.run
