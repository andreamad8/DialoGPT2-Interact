# DialoGPT2-Interact
Script to interact with the [DialoGPT](https://github.com/microsoft/DialoGPT) models. 


## How to use it? 
Clone the [DialoGPT](https://github.com/microsoft/DialoGPT) repo and add the interact.py into the main directory. Then run:
```
python interact.py --model_name_or_path ./models/medium --load_checkpoint ./models/medium/medium_ft.pkl --top_k 0
```
you can use both --top_k or --top_p. I played a bit and with --top_k 1 the model works quite well. You can also set the dialogue history length using --max_history, is you set it to -2 it will just use the last utterance. 


## Disclaimer
This is third-party reproduction of the decoding script. 
