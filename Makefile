train.build:
	# Build the docker image for training models
	docker image build \
		-t transfer \
		transferEngine/

train.run:
	# Run the training script
	docker run \
		--gpus all \
		-it \
		-v ${PWD}/transferEngine:/transferEngine \
		transfer \
		python main.py

train:
	# Build the docker image and run the training script
	make train.build
	make train.run
