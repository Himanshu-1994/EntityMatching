import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from torch.cuda.amp import autocast, GradScaler

from .dataset_gpt2 import DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification, GPT2Config
from tensorboardX import SummaryWriter
#from apex import amp
from transformers import AutoTokenizer

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         't5':'t5-base',
         'gpt2':'gpt2',
         'gpt2_medium':'gpt2-medium',
         'gpt2_large':'gpt2-large'}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()

        labs = [0,1]
        model_config = GPT2Config.from_pretrained(
            pretrained_model_name_or_path=lm_mp[lm],
            num_labels=2,
            return_dict=True,
            )
    
        self.model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=lm_mp[lm], 
            config=model_config, 
            ignore_mismatched_sizes=True,
            )
    
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
       
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens_dict = {'additional_special_tokens': ['[SEP]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))   
        self.model.config.pad_token_id = self.model.config.eos_token_id

        #special_tokens_dict = {'additional_special_tokens': ['COL', 'VAL']}
        #num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.model.config.hidden_size
        self.model.to(self.device)

    def forward(self, x1, y, mode='train'):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1,x1_mask,y = x1[0].to(self.device),x1[1].to(self.device),y.to(self.device)
        
        if mode == 'train':
             # (batch_size, seq_len)
            out = self.model(x1, token_type_ids=None, attention_mask=x1_mask,labels=y)
        else:
            out = self.model(x1, token_type_ids=None, attention_mask=x1_mask,labels=y)

        return out # .squeeze() # .sigmoid()


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x,x_m, y = batch
            
            out = model([x,x_m],y,'test')
            logits = out['logits']
            #print('logits shape ',logits.shape)
            probs = logits.softmax(dim=-1)[:, 1]
            #print('probs shape ',probs.shape)
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        report = None
        #report = metrics.classification_report(all_y, pred, labels=['negative','positive'])
        return f1, report
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    accumulation_steps = 1
    optimizer.zero_grad()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        
        x, x_mask, y = batch
        out = model([x,x_mask],y)
        loss = out['loss']/accumulation_steps
        #logits = out.logits

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug)
    model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=hp.lr)
    scaler = None
    if hp.fp16:
        #model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        scaler = GradScaler()
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    #writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    best_test_report = None
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1, test_report = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            best_test_report = test_report
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)
                with open(os.path.join(hp.logdir, 'report'),'w') as f:
                    f.write("Epoch:{}".format(epoch))
                    #f.write(best_test_report)
                    f.write(f"\n\nepoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        #scalars = {'f1': dev_f1,
        #           't_f1': test_f1}
        #writer.add_scalars(run_tag, scalars, epoch)

    #writer.close()
