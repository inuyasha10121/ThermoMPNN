import numpy as np
import pandas as pd
import os
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from Bio.PDB import PDBParser

from .utils.datasets import Mutation
from .utils.protein_mpnn_utils import alt_parse_PDB
from .thermompnn_benchmarking import get_trained_model
from .SSM import get_ssm_mutations


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def get_chains(pdb):
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('', pdb)
  chains = [c.id for c in structure.get_chains()]
  return chains

def run_inference_prediction(cfg: DictConfig, model_path: str, pdb_path: str, chain: str = 'A') -> pd.DataFrame:
    """Run scanning saturation mutagenesis on PDB
    
    Arguments:
        cfg: Configuration 
        model_path: Path to model checkpoint to load
        pdb_path: Path to PDB file to study
        chain: Chain of protein to scan

    Returns:
        Dataframe of mutation results
    """

    # define config for model loading
    #TODO: Eventually, have a config file for each model to avoid this being hardcoded
    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        }
    }

    cfg = OmegaConf.merge(config, cfg)

    # load the chosen model and dataset
    models = {
        "ThermoMPNN": get_trained_model(model_name=model_path,
                                        config=cfg, override_custom=True)
    }

    input_pdb = pdb_path
    pdb_id = os.path.basename(input_pdb).rstrip('.pdb')

    datasets = {
        pdb_id: pdb_path
    }

    raw_pred_df = pd.DataFrame(columns=['Model', 'Dataset', 'ddG_pred', 'position', 'wildtype', 'mutation'])
    row = 0
    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            if len(chain) < 1:  # if unspecified, take first chain
                chain = get_chains(input_pdb)[0]
            else:
                chain = chain
            mut_pdb = alt_parse_PDB(input_pdb, chain)
            mutation_list = get_ssm_mutations(mut_pdb[0])
            final_mutation_list = []

            # build into list of Mutation objects
            for n, m in enumerate(mutation_list):
                if m is None:
                    final_mutation_list.append(None)
                    continue
                m = m.strip()  # clear whitespace
                wtAA, position, mutAA = str(m[0]), int(str(m[1:-1])), str(m[-1])

                assert wtAA in ALPHABET, f"Wild type residue {wtAA} invalid, please try again with one of the following options: {ALPHABET}"
                assert mutAA in ALPHABET, f"Wild type residue {mutAA} invalid, please try again with one of the following options: {ALPHABET}"
                mutation_obj = Mutation(position=position, wildtype=wtAA, mutation=mutAA,
                                        ddG=None, pdb=mut_pdb[0]['name'])
                final_mutation_list.append(mutation_obj)

            #Run model prediction
            with torch.no_grad():
                pred, _ = model(mut_pdb, final_mutation_list)

            #Append results to dataframe
            num_results = len(final_mutation_list)

            ddG = torch.hstack([x['ddG'] for x in pred]).cpu().numpy()
            pos = np.zeros(num_results, dtype=np.uint16)
            wildtype = np.zeros(num_results, dtype='<U1')
            mutation = np.zeros(num_results, dtype='<U1')

            mask = np.ones(num_results, dtype=np.bool_)

            for i, mut in enumerate(final_mutation_list):
                if final_mutation_list is not None: #Not sure why this check is done, but keeping it for now
                    pos[i] = mut.position
                    wildtype[i] = mut.wildtype
                    mutation[i] = mut.mutation
                else:
                    mask[i] = False
            df = pd.DataFrame(columns=raw_pred_df.columns)
            df['ddG_pred'] = ddG[mask]
            df['position'] = pos[mask]
            df['wildtype'] = wildtype[mask]
            df['mutation'] = mutation[mask]
            df['Model'] = name
            df['Dataset'] = dataset_name

            raw_pred_df = pd.concat([raw_pred_df, df])
    
    return raw_pred_df