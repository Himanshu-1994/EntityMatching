import torch

from torch.utils import data
from transformers import AutoTokenizer,GPT2Tokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         't5':'t5-base',
         'gpt2':'gpt2',
         'gpt2_medium':'gpt2-medium',
         'gpt2_large':'gpt2-large'}

def get_tokenizer(lm):
    
    return GPT2Tokenizer.from_pretrained(
      pretrained_model_name_or_path=lm_mp[lm])

class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):
        self.lm = lm
        self.tokenizer = get_tokenizer(lm)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens_dict = {'additional_special_tokens': ['[SEP]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path,encoding='UTF-8')

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]

        oversample = False
        if oversample and da is not None:
            pos_pairs = []
            pos_labels = []
        
            for i,lab in enumerate(self.labels):
                if lab==1:
                  pos_pairs.append(self.pairs[i])
                  pos_labels.append(1)
            
            ovratio = 2
        
            self.pairs += pos_pairs*ovratio
            self.labels += pos_labels*ovratio

        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # augment if da is set
        if self.da is not None:
          sent = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
        else:
          sent = left + ' [SEP] ' + right
        # left + right
        
        x = self.tokenizer.encode_plus(sent,
                                      add_special_tokens=True,
                                      max_length=self.max_len,
                                      pad_to_max_length=True,
                                      return_attention_mask=True,
                                      truncation=True)

        return x['input_ids'],x['attention_mask'], self.labels[idx]
    
    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """

        x12, x12_mask, y = zip(*batch)
        return torch.LongTensor(x12), \
              torch.LongTensor(x12_mask), \
              torch.LongTensor(y)