import collections
import csv
from pathlib import Path

import numpy as np
import pd
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# TODO Task 1b - Implement LesionDataset
class LesionDataset(Dataset):
    #The __init__ function should have the following prototype
    #def __init__(self, img_dir, labels_fname):
    #img_dir is the directory path with all the image files
    #labels_fname is the csv file with image ids and their corresponding labels
    
    def __init__(self, img_dir, labels_fname):
        # TODO: Store the images and labels as member variables
        #       self.images and self.labels respectively.
        
        # SOLUTION LINE
        self.df = pd.read_csv(labels_fname)
        self.labels = np.argmax(np.array(df.iloc[:, 1:]), axis=1)
   

    def __len__(self):
        # TODO: Return the length (number of images) of the dataset
        # SOLUTION LINE
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
          img = Image.open(os.path.join(self.img_dir, self.df.iloc[idx, 0] + ".jpg"))
          image_2_npArray_2_tensor = transformstransforms.ToTensor()(np.asarray(img))
          
          # SOLUTION LINE
          image = self.images[idx]
          # SOLUTION LINE
          label = self.labels[idx]

          return image, label
# TODO Task 1e - Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):



# TODO Task 2b - Implement TextDataset
#               The __init__ function should have the following prototype
#                   def __init__(self, fname, sentence_len)
#                   - fname is the filename of the cvs file that contains each
#                     news headlines text and its corresponding label.
#                   - sentence_len the maximum sentence length you want the
#                     tokenized to return. Any sentence longer than that should
#                     be truncated by the tokenizer. Any shorter sentence should
#                     padded by the tokenizer.
#                We will be using the pretrained 'distilbert-base-uncased' transform,
#                so please use the appropriate tokenizer for it. NOTE: You will need
#                to include the relevant import statement.


class TextDataset(Dataset):
    
    def __init__(self, fname, sentence_len):
        
        # - fname is the filename of the cvs file that contains each
        #       news headlines text and its corresponding label.
        # - sentence_len the maximum sentence length you want the
        #       tokenized to return. Any sentence longer than that should
        #       be truncated by the tokenizer. Any shorter sentence should
        #       padded by the tokenizer.
        # TODO: set the tokenizer by using the AutoTokenizer to get 
        #       a tokenizer for the pretrained distilbert-base-uncased
        #       transformer
        
        df = pd.read_csv(fname)
        texts = df[2].str.slice(0, sentence_len).tolist()
        labels = df[0].tolist()
        
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        # Store the number of words in the tokenizer's vocabulary
        self.vocab_size = tokenizer.vocab_size
        # TODO: Call tokenizer with texts as an argument, setting the arguments
        # truncation and padding to True
        # tokens = ...
        # SOLUTION LINE
        tokens = tokenizer(texts, truncation=True, padding=True)

        # tokens is actually a dictionary with a few entries. We only want the
        # word IDs, so on the next line you should store the "input_ids" entry
        # in self.tokens
        # You should keep in mind that 'tokens' also has an "attention_mask" 
        # entry. In this lab we will not be using attention, which is why we 
        # won't be storing it.
        # self.tokens = ...
        # SOLUTION LINE
        self.tokens = tokens["input_ids"]
        
        self.labels = labels
   

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Here we initialise a tensor for the inputs at the given index
        inputs = torch.tensor(self.tokens[idx], device=self.device)

        # TODO: Initialise a tensor for the label at the given index
        # label = ...
        label = torch.tensor(self.labels[idx], device=self.device)

        return inputs, label
