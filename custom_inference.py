if __name__ == "__main__":
    from omegaconf import OmegaConf
    import torch
    from run.custom_inference import run_inference_prediction
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to model to use for inference.', required=True)
    parser.add_argument('--pdb', type=str, help='Input PDB to use for custom inference', required=True)
    parser.add_argument('--chain', type=str, default='A', help='Chain in input PDB to use.')
    parser.add_argument('--config', type=str, help='Path to configuration file.', required=True)

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    with torch.no_grad():
        results = run_inference_prediction(cfg, args.model_path, args.pdb, args.chain)
        ind = args.pdb.rfind('/')
        if ind == -1:
            ind = 0
        results.to_csv(f"./ThermoMPNN_inference_{args.pdb[ind+1:args.pdb.rfind('.')]}_results.csv")
