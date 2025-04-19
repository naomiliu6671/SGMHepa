import torch
import torch.nn as nn
from model_set.transformer import TransformerEncoder

torch.backends.cudnn.enabled = False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Classifier(nn.Module):

    def __init__(self, latent_size, mito_size, device, mito):
        super(Classifier, self).__init__()
        if mito:
            self.latent_size = latent_size + mito_size
        else:
            self.latent_size = latent_size
        self.classifier = nn.ModuleList([nn.Linear(self.latent_size, 16),
                                         nn.ReLU(),
                                         nn.Linear(16, 2)])
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        for m in self.classifier:
            x = m(x)
        return x


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_size, latent_size, device,
                 dropout=0.1):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout = nn.Dropout(dropout)
        self.encoder_type = 'trans'
        self.transformer = TransformerEncoder(num_layers=4,
                                              d_model=input_dim,
                                              dropout=dropout,
                                              layer_type='self')
        self.apply(weights_init)
        self.to(device)

    def forward(self, x, x_mask=None):
        x_t = x.transpose(1, 0)
        outputs_ = self.transformer(x_t, mask=x_mask)
        outputs_ = outputs_.transpose(1, 0)
        if x_mask is not None:
            outputs_ = outputs_ * x_mask[:, :, None]
        encoder_outputs = outputs_

        return encoder_outputs
