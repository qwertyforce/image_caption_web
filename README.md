# image_caption_web
OFA (https://github.com/ofa-sys/ofa) is used for image captioning
Supported operations: get_image_caption
```bash
pip install -r requirements.txt
```   
```
# do not git clone in image_caption_web folder
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --use-feature=in-tree-build ./

```



```pip3 install -r requirements.txt```  
You should install torch yourself https://pytorch.org/get-started/locally/.


```generate_captions.py ./path_to_img_folder``` -> generates captions, places them in id_caption.txt  
```image_caption_web.py``` -> web microservice   
