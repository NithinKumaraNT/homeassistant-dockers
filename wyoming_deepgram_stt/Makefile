.PHONY: local run update

VERSION := 0.2.0
TAG := wyoming-deepgram-stt
PLATFORMS := linux/amd64,linux/arm64,linux/arm/v7
MODEL := "nova-2"
HOST := 0.0.0.0
PORT := 10000

# all:
# 	docker buildx build . --platform "$(PLATFORMS)" --tag "$(TAG):$(VERSION)" --push

# update:
# 	docker buildx build . --platform "$(PLATFORMS)" --tag "$(TAG):latest" --push

build:
	docker build . -t "$(TAG):$(VERSION)" --build-arg TARGETARCH=amd64

run:
	docker run -it -p '$(PORT):$(PORT)' "$(TAG):$(VERSION)" --model "$(MODEL)" --uri 'tcp://$(HOST):$(PORT)' --debug