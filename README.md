# Video Captioning Project
## Overview
This project focuses on generating captions for videos using BLIP model. 

## Installation
To install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Run App
```bash
flask run
```
## Deployed System
You can access the deployed system [here](https://video-caption-4qny.onrender.com/).

## Build the image
```bash
docker build -t video-caption-generator .
```

## run the container
```bash
docker run -p 5000:5000 video-caption-generator
```
