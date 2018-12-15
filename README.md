# Neural Machine Translation
This project builds different neural machine translation system for translating Vietnamese to English and Chinese to English. 

### Models
  - Recurrent neural network based encoder-decoder without attention
  - Recurrent neural network based encoder-decoder with attention
  - Replace the recurrent encoder with either convolutional or self-attention based encoder
  - (Optional) Build either or both fully self-attention translation system or/and multilingual translation system

### Run Training
```sh
$ python main.py --encoder 'bigru' --decoder 'gruattn' --n_directions 2 --use_attn 1 --print_freq 10
```

### Contributors
Jiayi Gao
Sylvie Shao
Wei Zhang
