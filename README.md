This is the source "Speech To Text VietNamese and Japanese".  

# Install 

python 3.8.13
```
pip install -r requirements.txt
```

# Run API

At server, activate enviroment "ekyc"

```
conda activate ekyc
python app.py
```

# Speech To Text Japanese
- Read more [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi) by [VOSK](https://alphacephei.com/vosk/)

# Speech To Text VietNamese

## Architecture
<img width="946" alt="image" src="./docs/wave2vec2.0.png">

Use [WAVE2VEC2.0](https://arxiv.org/abs/2006.11477), pretrain [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)

## Training

### 1. Speech to Text
At server, activate enviroment "hoang_speech"
```
conda activate hoang_speech
```
### 1.1 Data Preparation

- Download data at [driver](https://drive.google.com/file/d/1UaV-p2KKTSN6mGzuquGpjlfnxGL1qPen/view?usp=sharing)

- Unzip and put in folder ./training/wave2vec_vn

- Convert data to format of library datasets huggingface
    ```
        cd ./training/wave2vec_vn
        python create_dataset.py --input_dir ./s2t_data/audios  --output_dir ./s2t_data/s2t_vn
    ``` 
### 1.2. Training

```
python train.py --dataset ./s2t_data/s2t_vn --output wav2vec2_s2t_vn
```  

### 1.3. Inference
```
python test --repo_name wav2vec2_s2t_vn
```  

### 2. Add Punctuation
At server, activate enviroment "speech2text"

```
conda activate speech2text
```

### 2.1 Data Preparation
- Source code training: s2t_rec/punc_restore_vn

- Download data at [driver](https://drive.google.com/drive/folders/1cdWlgkmGysf69uZKfYdFqc7BC53NnfmC?usp=sharing)

- If create new data, create 4 folders contain train, valid, test and punc_vocab (contain punctation such as . , ! ?). Note that sentence must split follow token.  
    Ex: "Nhưng không bao_giờ có_thể quay ngược"

### 2.2 Training

```
python train.py 
```  
**Note:** Weights trained is stored at [driver](https://drive.google.com/drive/folders/1ZQ9L1f4KzpTvlcWXn3ZJOXqSZhEzghaG?usp=sharing). Download and put it in ./s2t_rec/punc_restore_vn/weights
