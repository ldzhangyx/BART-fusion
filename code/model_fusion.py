import torch
from torch import nn
from transformers import BartTokenizer
from modeling_bart import BartForMultimodalGeneration
from music_encoder import CNNSA



class CommentGenerator_fusion(nn.Module):
    def __init__(self):
        super(CommentGenerator_fusion, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model_path = "best_model.pth"
        self.music_encoder = CNNSA().cuda()
        self.music_encoder.load_state_dict(torch.load(model_path))
        # trial: fix music encoder's params
        for params in self.music_encoder.parameters():
            params.requires_grad = False

        self.bart = BartForMultimodalGeneration.from_pretrained("facebook/bart-base",
                                                                fusion_layers=[4,5], # [4,5]
                                                                use_forget_gate=False, # [True]
                                                                dim_common=768, # 256
                                                                n_attn_heads=1).cuda()


    def forward(self, input_sentence_list, music_ids, labels=None):
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
        music_features = self.music_encoder(music_ids)
        output = self.bart(input_ids=encoded_input['input_ids'].cuda(),
                           attention_mask=encoded_input['attention_mask'].cuda(),
                           labels=labels['input_ids'].cuda(),
                           music_features=music_features
                           # labels
                           )
        return output

    def generate(self, input_sentence_list, music_ids, is_cuda=True):
        encoded_input = self.tokenizer(input_sentence_list,
                                       padding=True,
                                       truncation=True,
                                       return_tensors='pt',
                                       )
        music_features = self.music_encoder(music_ids)
        output_ids = self.bart.generate(encoded_input['input_ids'].cuda(),
                                        num_beams=5,
                                        max_length=512,
                                        early_stopping=True,
                                        do_sample=True,
                                        music_features=music_features)
        return ([self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for g in output_ids])
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# encoded_input = tokenizer(['Hello all', 'Hi all'], return_tensors='pt')
# print(encoded_input) 