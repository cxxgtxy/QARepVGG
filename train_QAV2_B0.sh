sh dist_train.sh QARepVGGV2-B0 8  --data-path /workdir/ILSVRC2012/ --batch-size 128  --dist-eval --tag qa --wd 4e-5 --no-model-ema &> QARepVGGV2-B0.log
