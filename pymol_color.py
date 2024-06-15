import pymol
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd

def color_residues_by_value(pdb_path, output_path, base_folder='dataset'):
    # Step 1: Read data file and record residue numbers
    # Load predictions from .npy file
    data = np.load(f'{output_path}-preds.npy')
    values = data[0]

    # Step 2: Load PDB file and create object
    pymol.cmd.load(f'{pdb_path}', 'protein')
    pymol.cmd.show_as('cartoon', 'protein')
    #pymol.cmd.show('lines', 'protein')
    pymol.cmd.bg_color('white')
    pymol.cmd.color('gray', 'protein')

    # Step 3: Get residue numbers and information from PyMOL
    all_residues_info = pymol.cmd.get_model('protein', 1)
    all_residues_nums = [int(atom.resi) for atom in all_residues_info.atom]
    all_residues_nums = sorted(set(all_residues_nums))

    # Step 4: Generate colors based on values using 'bwr' colormap
    norm = Normalize(vmin=0.2, vmax=1.1)  # Adjust vmin and vmax to change blue proportion
    cmap = plt.cm.get_cmap('bwr')
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = [sm.to_rgba(value)[:-1] for value in values]  # Convert RGBA to RGB

    # Step 5: Apply colors to residues
    for idx, color in enumerate(colors):
        pymol.cmd.set_color(f'mycol{idx}', color)
        pymol.cmd.color(f'mycol{idx}', f'resi {all_residues_nums[idx]}')

    pymol.cmd.set('ray_trace_fog', 0)  # 关闭景深效果
    pymol.cmd.zoom('all')  # 放大到合适的位置
    

    # Step 7: Save image
    image_path = f'{pdb_path}-colored.png'
    pymol.cmd.png(image_path, dpi=300, ray=1, width=800, height=600)



# Example usage
if __name__ == '__main__':
    ## Read file
    pdb_path = 'example/gsdmd_swmodel.pdb'
    output_path = 'example/predict'

    pymol.finish_launching(['pymol', '-cq'])
    color_residues_by_value(pdb_path, output_path)
    pymol.cmd.delete('all')
    


    
