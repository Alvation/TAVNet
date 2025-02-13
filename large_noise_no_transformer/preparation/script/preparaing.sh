

TEXT="/workspace/dzy/AVTSR_Fairseq/AVTNet/data/lrs3/433h_data"


fairseq-preprocess --srcdict /workspace/dzy/AVTSR_Fairseq/AVTNet/data/lrs3/433h_data/dict.bert.txt \
    --trainpref $TEXT/train.wrd --validpref $TEXT/valid.wrd --testpref $TEXT/test.wrd \
    --destdir $TEXT --only-source --dict-only