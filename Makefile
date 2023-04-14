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
		python train_model.py \
			--epochs 100 \
			--split 0.2 \
			--batch-size 16 \
			--dataset-path trainingImages/ \
			--target-shape 28 28 3 \
			--model-path model.tf

train:
	# Build the docker image and run the training script
	make train.build
	make train.run
