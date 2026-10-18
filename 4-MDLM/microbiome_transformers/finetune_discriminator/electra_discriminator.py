from transformers import ElectraConfig,ElectraForSequenceClassification
import torch.nn as nn
import torch
import pdb

class ElectraDiscriminator(nn.Module):

    def __init__(self,config:ElectraConfig,embeddings,discriminator = None, embed_layer = None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size,padding_idx = config.vocab_size-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)
        if discriminator:
            self.discriminator = ElectraForSequenceClassification.from_pretrained(discriminator,config=config)
        else:
            self.discriminator = ElectraForSequenceClassification(config)
        self.softmax = nn.Softmax(1)

    # Note the 4 spaces of indentation here!
    def forward(self,data,attention_mask,labels):
        data = self.embed_layer(data)
        output = self.discriminator(attention_mask=attention_mask,inputs_embeds=data,labels=labels)
        
        # FIX: Return raw logits for training loss calculation, but keep softmax available if needed elsewhere
        logits = output['logits']
        scores = self.softmax(logits)
        loss = output['loss']
        
        # We MUST return the raw logits as 'scores' so CrossEntropyLoss works correctly.
        return loss, logits