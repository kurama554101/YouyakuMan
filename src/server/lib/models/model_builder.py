import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from torch.nn.init import xavier_uniform_

from lib.models.encoder import TransformerInterEncoder
from lib.models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.learning_rate, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        optim.learning_rate = args.learning_rate
        for param_group in optim.optimizer.param_groups:
            param_group['lr'] = args.learning_rate

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Bert(nn.Module):
    def __init__(self, model_path):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(model_path)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask=mask)
        top_vec = encoded_layers[-1]
        return top_vec


class SummarizerParams:
    def __init__(self, device, pretrained_model_path):
        self.device = device
        self.pretrained_model_path = pretrained_model_path
        self.ff_size               = 2048 # dimention size of feed forward hidden layer in BERT 
        self.heads                 = 8    # attention head counts
        self.dropout               = 0.1  # dropout ratio
        self.inter_layers          = 2    # layer counts of fine-tuning Transformer encoder layer
        self.param_init            = 0.0  # ??
        self.param_init_glorot     = True # ??

    @classmethod
    def from_args(cls, args):
        instance = cls(device=args.device, pretrained_model_path=args.pretrained_model_path)
        instance.ff_size           = args.ff_size
        instance.heads             = args.heads
        instance.dropout           = args.dropout
        instance.inter_layers      = args.inter_layers
        instance.param_init        = args.param_init
        instance.param_init_glorot = args.param_init_glorot
        return instance


class Summarizer(nn.Module):
    def __init__(self, params:SummarizerParams):
        super(Summarizer, self).__init__()
        self.device = params.device
        self.bert = Bert(params.pretrained_model_path)
        self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size,
                                               params.ff_size, params.heads,
                                               params.dropout, params.inter_layers)
        self.h_params = params

        if params.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-params.param_init, params.param_init)

        if params.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device=self.device)

    def load_cp(self, pt):
        self.load_state_dict(pt, strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):

        top_vec = self.bert(x, segs, mask)
        
        # cls部分のベクトルを抽出している(= clsは各センテンスごとに文頭に置かれるため、文頭のCLSをその文全体の特徴量としてみなしている)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
