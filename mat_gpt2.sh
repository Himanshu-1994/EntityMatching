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
cp -r /staging/pandotra/gpt2_wdc_all_title_xlarge/* ./checkpoints
# cp -r nltk_data ..
nvidia-smi
echo "Run!"
mkdir output_gpt2

python matcher_gpt2.py \
  --task wdc_all_title_xlarge \
  --input_path data/er_magellan/Structured/Walmart-Amazon/test.txt \
  --output_path output_gpt2/output_Structured_Walmart-Amazon-test.jsonl \
  --lm gpt2 \
  --max_len 256 \
  --use_gpu \
  --checkpoint_path checkpoints/

cp -r output_gpt2 /staging/pandotra/gpt2_wdc_all_title_xlarge/
rm -r checkpoints
