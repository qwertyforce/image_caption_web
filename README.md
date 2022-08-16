# image_caption_web
OFA (https://github.com/ofa-sys/ofa) is used for image captioning
```bash
pip install -r requirements.txt
wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt
mv caption_large_best_clean.pt ./checkpoints/caption.pt

git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --use-feature=in-tree-build ./

```
