import torch
from torch import nn
from transformers import BartTokenizer, BartForConditionalGeneration


class CommentGenerator(nn.Module):
    def __init__(self):
        super(CommentGenerator, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # self.bart_config = BartConfig()
        self.condition = None


    def forward(self, input_sentence_list, labels=None):
        encoded_input = self.tokenizer(
            input_sentence_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        if labels is not None:
            labels = self.tokenizer(
                labels,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            )
        output = self.bart(input_ids=encoded_input['input_ids'].cuda(),
                           attention_mask=encoded_input['attention_mask'].cuda(),
                           labels=labels['input_ids'].cuda(),
                           # labels
                           )
        return output

    def generate(self, input_sentence_list, is_cuda=True):
        encoded_input = self.tokenizer(input_sentence_list,
                                       padding=True,
                                       truncation=True,
                                       return_tensors='pt',
                                       )
        output_ids = self.bart.generate(encoded_input['input_ids'].cuda(),
                                        num_beams=4,
                                        max_length=512,
                                        early_stopping=True,
                                        do_sample=True)
        return ([self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for g in output_ids])
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# encoded_input = tokenizer(['Hello all', 'Hi all'], return_tensors='pt')
# print(encoded_input)



