# Bark-Voice-Cloning

## Repository Description

This repository is a derivative of some other Bark-Cloning implementations. It builds upon their work and incorporates additional features and modifications specific to this project.


## Installing

- Conda:
```
conda create -n bark-env python=3.10.10
conda activate bark-env
pip install -r requirements.txt
```

- Docker:
```
docker build -t bark-voice-clone .
docker run -t --gpus all --name bark-env --restart always bark-voice-clone:latest
docker exec -itd bark-env bash
pip install -r requirements.txt 
```

## Pineline

- To generate npz:
```
python3 main.py --scenario generate_npz
```

- To clone voice:
```
python3 main.py --scenario clone_voice
```

### Todo
- [ ] Prepare Vietnamese data
- [ ] Traning Hubert with Vietnamese data
- [ ] Modify params for beter results