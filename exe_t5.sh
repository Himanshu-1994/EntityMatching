#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e
# cp /staging/pandotra/env3.tar.gz ./
# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=env3
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
export HOME=$(pwd)
echo 'Home Path given'
echo $HOME
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

echo "Hello!"
tar -xvf ditto.tar
# modify this line to run your desired Python script and any other work you need to do
#export NLTK_DATA=~/ditto/
#source ~/.bashrc

cd ditto
# cp -r nltk_data ..
nvidia-smi
echo "Run!"

python train_ditto_t5.py \
  --task wdc_all_title_xlarge \
  --batch_size 32 \
  --max_len 256 \
  --lr 2e-5 \
  --n_epochs 20 \
  --finetuning \
  --fp16 \
  --lm t5 \
  --da drop_col \
  --summarize \
  --save_model

cp -r checkpoints /staging/pandotra/t5_base_wdc_all_title_xlarge/
rm -r checkpoints
