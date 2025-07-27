import torch
import torch_geometric as pyg
import torch.nn.functional as F
import pickle

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

torch.manual_seed(0) # Para reproduzir os resultados

# torch.set_default_tensor_type(torch.DoubleTensor)


class DotProductPredictor(torch.nn.Module):
    def forward(self, x, edge_index):
        src, dst = edge_index
        score = (x[src] * x[dst]).sum(dim=-1)
        return score
    

class CVRP_Base(torch.nn.Module):
    def __init__(self, act = "relu"):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(pyg.nn.SimpleConv())
        self.act = "relu"
        self.pred = DotProductPredictor()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        for layer in self.layers:
            x = layer(x,edge_index,edge_attr)
            if self.act == "relu":
                x = F.relu(x)
        x = self.pred(x, data.edge_index)
        return x
    
class CVRP_ResGatedGraphConv(CVRP_Base):
    def __init__(self, n_layers, embedding_dim, act="relu"):
        super().__init__(act)

        for i in range(1,n_layers+1):
            in_feats = -1 if i==1 else embedding_dim
            conv = pyg.nn.ResGatedGraphConv(
                in_channels=in_feats, out_channels=embedding_dim, aggr='mean',edge_dim=1)
            self.layers.append(conv)
            
class CVRP_GraphConv(CVRP_Base):
    def __init__(self, n_layers, embedding_dim, act="relu"):
        super().__init__(act)

        for i in range(1,n_layers+1):
            in_feats = -1 if i==1 else embedding_dim
            conv = pyg.nn.GraphConv(
                in_channels=in_feats, out_channels=embedding_dim, aggr='mean')
            self.layers.append(conv)
    
class CVRP_GCNConv(CVRP_Base):
    def __init__(self, n_layers, embedding_dim, act="relu"):
        super().__init__(act)

        for i in range(1,n_layers+1):
            in_feats = -1 if i==1 else embedding_dim
            conv = pyg.nn.GCNConv(
                in_channels=in_feats, out_channels=embedding_dim, aggr='mean')
            self.layers.append(conv)
        
class PYG_DeepGCNEncoder(torch.nn.Module):
    def __init__(
        self,
        n_layers=1,
        embedding_dim=128,
        input_h_dim=1,
        input_e_dim=1,
        conv_type="gen_conv",
        n_heads=8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv_type = conv_type

        self.node_encoder = torch.nn.Linear(input_h_dim, embedding_dim)
        self.edge_encoder = torch.nn.Linear(input_e_dim, embedding_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(1, n_layers + 1):
            if conv_type == "gen_conv":
                conv = pyg.nn.GENConv(
                    embedding_dim,
                    embedding_dim,
                    aggr="softmax",
                    t=1.0,
                    learn_t=True,
                    norm="layer",
                )  
            elif conv_type == "transformer_conv":
                conv = pyg.nn.TransformerConv(
                        embedding_dim,
                        embedding_dim // n_heads,
                        heads=n_heads,
                        edge_dim=embedding_dim,
                )
            elif conv_type == "pdn_conv":
                conv = pyg.nn.PDNConv(
                    embedding_dim, #in_channels
                    embedding_dim, #out_channels
                    embedding_dim, #edge_dim
                    embedding_dim) #hidden_edge_dim
            norm = torch.nn.BatchNorm1d(embedding_dim)
            act = torch.nn.ReLU(inplace=True)
            layer = pyg.nn.DeepGCNLayer(conv, norm, act, block="res+")
            self.layers.append(layer)
        self.pred = DotProductPredictor()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        x = self.pred(x, data.edge_index)
        return x
    


def get_model (model_name, model_subsection, n_layers, n_nodes):

    if model_name == "ResGatedGraphConv":
        model = CVRP_ResGatedGraphConv(n_layers=n_layers, embedding_dim=n_nodes)

    elif model_name == "GCNConv":
        model = CVRP_GCNConv(n_layers=n_layers, embedding_dim=n_nodes)
        
    elif model_name == "GraphConv":
        model = CVRP_GraphConv(n_layers=n_layers, embedding_dim=n_nodes)
        
    elif model_name == "DeepGCNEncoder":
        if model_subsection == "gen-conv":
            model = PYG_DeepGCNEncoder(n_layers=n_layers, embedding_dim=n_nodes, conv_type="gen_conv")
            
        elif model_subsection == "pdn-conv":
            model = PYG_DeepGCNEncoder(n_layers=n_layers, embedding_dim=n_nodes, conv_type="pdn_conv")
            
        elif model_subsection == "transformer-conv":
            model = PYG_DeepGCNEncoder(n_layers=n_layers, embedding_dim=n_nodes, conv_type="transformer_conv")
        
    else:
        print("Error in setting a model")
        return None

    return model


def get_model_parameters (model_description):
    parameters = model_description.split("_")
    del parameters[0]
    parameters[2] = int(parameters[2])
    parameters[3] = int(parameters[3])

    return parameters