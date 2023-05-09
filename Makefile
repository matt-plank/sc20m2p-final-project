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

workspace:
	# Run the example
	make base.build
	docker run \
		--gpus all \
		-it \
		--volume $(shell pwd)/transferEngine:/transferEngine \
		transfer-train \
		/bin/bash
