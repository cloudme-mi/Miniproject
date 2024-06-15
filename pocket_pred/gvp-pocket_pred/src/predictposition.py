import tensorflow as tf
from models import MQAModel
import numpy as np
from glob import glob
import mdtraj as md
import os

from validate_performance_on_xtals import process_strucs, predict_on_xtals

def make_predictions(pdb_paths, model, nn_path, debug=False, output_basename=None):
    '''
        pdb_paths : list of pdb paths
        model : MQAModel corresponding to network in nn_path
        nn_path : path to checkpoint files
    '''
    strucs = [md.load(s) for s in pdb_paths]
    X, S, mask = process_strucs(strucs)
    if debug:
        np.save(f'{output_basename}_X.npy', X)
        np.save(f'{output_basename}_S.npy', S)
        np.save(f'{output_basename}_mask.npy', mask)
    predictions = predict_on_xtals(model, nn_path, X, S, mask)
    return predictions


# main method
from Bio.PDB import PDBParser, PDBIO, Select

class ResidueSelect(Select):
    def __init__(self, residues):
        self.residues = residues
    def accept_residue(self, residue):
        return residue in self.residues

def savepocket(pocket_centerid,strucs,output_folder):
    # Load the structure
    parser = PDBParser()
    structure = parser.get_structure('protein', strucs[0])

    # Get the residue
    residue_100 = list(structure.get_residues())[pocket_centerid[0]]  # 0-indexed

    # Find neighboring residues
    neighbors = []
    for residue in structure.get_residues():
        for atom in residue:
            if atom - residue_100['CA'] <= 10.0:  # Distance check
                neighbors.append(residue)
                break

    # Write to a new PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(f'{output_folder}/pocket.pdb', ResidueSelect([residue_100] + neighbors))

    


if __name__ == '__main__':
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Provide input pdb(s), output name, and output folder")

    # Add the arguments
    parser.add_argument('--strucs', nargs='+', required=True, help='The pdb files')
    parser.add_argument('--output_name', required=True, help='The output name')
    parser.add_argument('--output_folder', default='.', help='The output folder')

    # Parse the arguments
    args = parser.parse_args()

    strucs = args.strucs
    output_name = args.output_name
    output_folder = args.output_folder

    # debugging mode can be turned on to output protein features and sequence
    debug = False

    # Load MQA Model used for selected NN network
    nn_path = "./pocket_pred/gvp-pocket_pred/models/pocketminer"
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    
    
    if debug:
        output_basename = f'{output_folder}/{output_name}'
        predictions = make_predictions(strucs, model, nn_path, debug=True, output_basename=output_basename)
    else:
        predictions = make_predictions(strucs, model, nn_path)

    # output filename can be modified here
    np.save(f'{output_folder}/{output_name}-preds.npy', predictions)
    np.savetxt(os.path.join(output_folder,f'{output_name}-predictions.txt'), predictions, fmt='%.4g', delimiter='\n')
    
    # find pocket center amino acid position according to the predictions
    pocket_centerid = np.argmax(predictions, axis=1)

    savepocket(pocket_centerid,strucs,output_folder)



