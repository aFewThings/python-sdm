import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # RELU
        nn.init.zeros_(m.bias)
        print(m.__class__, ' initialized.')


# only supports environmental vector data
class SDM_DNN(nn.Module):
    def __init__(self, in_features=41, out_features=8, n_labels=1, drop_out=0, activation_func=nn.ReLU):
        super().__init__()
        self.n_labels = n_labels
        self.out_features = out_features
        self.activation_func = activation_func

        self.feature_extractor = nn.Sequential(
            self.linear_block(in_features, 16, drop_out),
            self.linear_block(16, 16, drop_out),
            self.linear_block(16, out_features, drop_out),
        )

        self.classifier = nn.Sequential(
            self.linear_block(8, 8),
            self.linear_block(8, 8),
            nn.Linear(8, n_labels),
        )

        self.feature_extractor.apply(init_weights)
        self.classifier.apply(init_weights)

    def linear_block(self, in_features, out_features, drop_out=0):
        layers = nn.Sequential(
            nn.Linear(in_features, out_features), 
            nn.Dropout(p=drop_out), 
            nn.LayerNorm(out_features),
            self.activation_func()
        )
        return layers

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features) # Nx1
        if self.n_labels == 1:
            logits = torch.squeeze(logits, dim=-1)
        if self.training is not True:
            logits = torch.sigmoid(logits)
        return logits
