import torch
import torch.nn as nn
from mil_models.tabular_snn import TabularSNN


class ABMIL_Fusion(nn.Module):
    def __init__(self, in_dim, clinical_in_dim, n_classes):
        super().__init__()

       
        self.embedding = nn.Linear(in_dim, 512)
        self.attention = nn.Linear(512, 1)

        
        self.tabular_net = TabularSNN(clinical_in_dim=clinical_in_dim)

       
        with torch.no_grad():
            dummy_input = torch.zeros(1, clinical_in_dim)
            clinical_out_dim = self.tabular_net(dummy_input).shape[1]

      
        self.classifier = nn.Linear(512 + clinical_out_dim, n_classes)

    def forward(self, x_bag, x_clinical):
        h = torch.tanh(self.embedding(x_bag))         
        a = torch.softmax(self.attention(h), dim=1)    
        z = torch.sum(a * h, dim=1)                    

        z_tab = self.tabular_net(x_clinical)           
        z_fusion = torch.cat((z,  z_tab), dim=-1)


        out = self.classifier(z_fusion)
        return {'logits': out, 'loss': None}


class ABMIL_Fusion_BN(nn.Module):
    def __init__(self, in_dim, clinical_in_dim, n_classes):
        super().__init__()

      
        self.embedding = nn.Linear(in_dim, 512)
        self.attention = nn.Linear(512, 1)

       
        self.tabular_net = TabularSNN(clinical_in_dim=clinical_in_dim)

       
        with torch.no_grad():
            dummy_input = torch.zeros(1, clinical_in_dim)
            clinical_out_dim = self.tabular_net(dummy_input).shape[1]

        self.bn_clinical = nn.BatchNorm1d(clinical_out_dim)
        self.classifier = nn.Linear(512 + clinical_out_dim, n_classes)

    def forward(self, x_bag, x_clinical):
        h = torch.tanh(self.embedding(x_bag))        
        a = torch.softmax(self.attention(h), dim=1)  
        z = torch.sum(a * h, dim=1)                

        z_tab = self.tabular_net(x_clinical)          
        z_tab = self.bn_clinical(z_tab)
        z_fusion = torch.cat((z, 0.2 * z_tab), dim=-1)

        out = self.classifier(z_fusion)
        return {'logits': out, 'loss': None}
