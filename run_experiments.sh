python main.py --data /data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/raw/ \
--save /data/nfsdata/nlp/projects/glyph_lm_cnn1/ --model LSTM --emsize 1500 --dropout 0.2 --gpu_id 3 \
--nfeat 300 --nhid 1500 --nlayers 2 --lr 0.02 --epochs 500 --batch_size 200 --bptt 35 \
--font_size 12 --font_path /data/nfsdata/nlp/fonts/Noto-hinted/NotoSansCJKsc-Regular.otf