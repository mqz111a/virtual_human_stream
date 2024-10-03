# Adopting the GPT vitamins scheme, BERT vitamins are suitable for long audio training, while GPT vitamins run short audio for fast inference
## Deploy TTS inference
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
git checkout fast_inference_
## 1. Install dependency libraries
```
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
From GPT SoVITS Models（ https://huggingface.co/lj1995/GPT-SoVITS ）Download pre trained models and place them in ` GP_SoVITS/GPTSoVITS/trainingd_models `

attention
```
It is to place the GPT SoVITS model file into the pretrained-models directory
```
as follows
```
pretrained_models/
--chinese-hubert-base
--chinese-roberta-wwm-ext-large
s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
s2D488k.pth
s2G488k.pth
```

## 3. Start up
### 3.1 Launching the WebUI Interface (for testing effectiveness)
python GPT_SoVITS/inference_webui.py

### 3.2 Starting API Services: 
python api_v3.py


## 4. interface specification 

### 4.1 Text-to-Speech

endpoint: `/tts`  
GET:
```
http://127.0.0.1:9880/tts?text=我是你最好的助手小可。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是哪吒三太子&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                                                 # str.(required) text to be synthesized
    "text_lang": "",                                            # str.(required) language of the text to be synthesized
    "ref_audio_path": "",                                       # str.(required) reference audio path.
    "prompt_text": "",                                          # str.(optional) prompt text for the reference audio
    "prompt_lang": "",                                          # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                                                 # int.(optional) top k sampling
    "top_p": 1,                                                 # float.(optional) top p sampling
    "temperature": 1,                                           # float.(optional) temperature for sampling
    "text_split_method": "cut5",                                # str.(optional) text split method, see text_segmentation_method.py for details.
    "batch_size": 1,                                            # int.(optional) batch size for inference
    "batch_threshold": 0.75,                                    # float.(optional) threshold for batch splitting.
    "split_bucket": true,                                       # bool.(optional) whether to split the batch into multiple buckets.
    "speed_factor":1.0,                                         # float.(optional) control the speed of the synthesized audio.
    "fragment_interval":0.3,                                    # float.(optional) to control the interval of the audio fragment.
    "seed": -1,                                                 # int.(optional) random seed for reproducibility.
    "media_type": "wav",                                        # str.(optional) media type of the output audio, support "wav", "raw", "ogg", "aac".
    "streaming_mode": false,                                    # bool.(optional) whether to return a streaming response.
    "parallel_infer": True,                                     # bool.(optional) whether to use parallel inference.
    "repetition_penalty": 1.35,                                 # float.(optional) repetition penalty for T2S model.
    "tts_infer_yaml_path": “GPT_SoVITS/configs/tts_infer.yaml”  # str.(optional) tts infer yaml path
}
```

## Deploy TTS training
https://github.com/RVC-Boss/GPT-SoVITS  
Switch to a self trained model 
### Switch GPT model

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/xxx.ckpt
```
RESP: 
Success: Return "success", HTTP code 200
Failed: Returned JSON with error message, HTTP code 400


### Switch Sovits model

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/xxx.pth
```

RESP: 
Success: Return "success", HTTP code 200
Failed: Returned JSON with error message, HTTP code 400
    
"""

##If you need to deploy using autodl
Please use https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS As a basic image, you can quickly deploy it
