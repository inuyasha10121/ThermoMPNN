import numpy as np
import torch
import torch.nn as nn
from .protein_mpnn_utils import ProteinMPNN, tied_featurize
from .model_utils import featurize
import os


HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True


def get_protein_mpnn(cfg, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings
    
    Arguments:
        cfg: Configuration information
        version: Version of model to load
    """
    hidden_dim = 128
    num_layers = 3 

    model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_weight_dir, version)
    # checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, k_neighbors=checkpoint['num_edges'], augment_eps=0.0)
    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if cfg.model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model


class TransferModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = get_protein_mpnn(cfg)
        EMBED_DIM = 128
        HIDDEN_DIM = 128

        hid_sizes = [ HIDDEN_DIM*self.num_final_layers + EMBED_DIM ]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        print('MLP HIDDEN SIZES:', hid_sizes)

        if self.lightattn:
            print('Enabled LightAttention')
            self.light_attention = LightAttention(embeddings_dim=HIDDEN_DIM*self.num_final_layers + EMBED_DIM )

        self.both_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)

    def forward(self, pdb, mutations, tied_feat=True):        
        #Get number of mutations
        num_mutations = len(mutations)

        #Get device to run everything on
        device = next(self.parameters()).device

        #Featurize input PDB
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        # Get ProteinMPNN structure embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)

        #Setup input space, modified by if final layers are present
        aa_index_mut = np.zeros(num_mutations, dtype=np.int16)
        aa_index_wt = np.zeros(num_mutations, dtype=np.int16)
        mut_positions = np.zeros(num_mutations, dtype=np.int16)
        for i in range(num_mutations):
            aa_index_mut[i] = ALPHABET.index(mutations[i].mutation)
            aa_index_wt[i] = ALPHABET.index(mutations[i].wildtype)
            mut_positions[i] = mutations[i].position
        
        mpnn_embed_size = mpnn_embed[0][0].shape[0]
        batch_indexer = np.arange(num_mutations)
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
            mpnn_hid_size = mpnn_hid[0][0].shape[0]
            input = torch.zeros((num_mutations, mpnn_embed_size + mpnn_hid_size, 1), dtype=torch.float32, device=device)
            mpnn_hid[0][mut_positions]
            input[batch_indexer, :mpnn_hid_size, 0] = mpnn_hid[0][mut_positions]
            input[batch_indexer, mpnn_hid_size:, 0] = mpnn_embed[0][mut_positions]

        else:
            input = torch.zeros((num_mutations, mpnn_embed_size, 1), dtype=torch.float32, device=device)
            input[batch_indexer, :, 0] = mpnn_embed[0][mut_positions]

        if self.lightattn:
            input = self.light_attention(input, None)

        both_input = self.both_out(input).unsqueeze(-1)
        ddg_out = self.ddg_out(both_input)
        if self.subtract_mut:
            ddg = ddg_out[batch_indexer,aa_index_mut] - ddg_out[batch_indexer, aa_index_wt]
        else:
            ddg = ddg_out[batch_indexer,aa_index_mut]
        out = [{"ddG": x} for x in ddg]
        return out, None


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)
