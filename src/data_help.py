import regex as re
import multiprocessing as mp
import sys
import os
import pickle

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdchem


if os.path.abspath('IFG') not in sys.path:
  sys.path.append(os.path.abspath('IFG'))
import ifg
from ifg.chem.molecule import Molecule

from vocab import FGVocab

    
regex = {
      'remove_tokens': r'\[(nH|Li|Mg|Pb|Bi|Si|B|OH|PH|I|Mn)[\d\+\-]*\]|B',
      'remove_tokens_lazy': r'\[(?!N[+-]?\])(?:[^\[\]]|(?R))*\]|B',
      'reaction_tokens': r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
  }

'''Using rdkit to convert to canonicalize smiles'''
def convert_smiles(smiles_mol, remove_all_elements = False):
  
  '''parse for rdkit'''
  mol  = "".join(
      re.findall(regex['reaction_tokens'], smiles_mol)
      )

  '''using rdkit'''
  mol = Chem.MolFromSmiles(mol) 
  for atom in mol.GetAtoms():  atom.SetIsotope(0)
  Chem.RemoveStereochemistry(mol)
  mol = Chem.RemoveHs(mol)
  mol = Chem.MolToSmiles(mol, isomericSmiles=False)


  '''parse for IFG'''
  mol = re.sub(
      pattern = regex['remove_tokens_lazy'] if remove_all_elements else regex['remove_tokens'] 
      , repl = '',string = mol, flags= re.IGNORECASE)
  
  return mol

'''Using ifg to find functional groups in molecule'''
def mol_fgs(smiles_molecule, first_attempt = True):
    fg_idx_list = []
    mol = convert_smiles(smiles_molecule) if first_attempt else convert_smiles(smiles_molecule, remove_all_elements= True) 
    
    try:
      mol = Molecule(mol, type = 'mol') 
      overlap_fgs, actual_fgs = mol.createFunctionalGroups()
      
      # if not first_attempt: print(f"Successfully retried with {mol}. Found: {actual_fgs.keys()}.")

      return [(fg, c) for fg,c in actual_fgs.items()]
      
    except Exception as e:
      if first_attempt:
        # print(f"Retrying on molecule : {mol}, preprocessing: {smiles_molecule}", end = ". ")
        return mol_fgs(smiles_molecule, first_attempt = False)

      else:
        # print(f"Could not parse molecule: {mol}, preprocessing: {smiles_molecule} \t Error message: {str(e)}.")
        return []
      
'''
Iterate over molecules in a reaction 
'''
def rxn_fgs(reaction):
  reactants, product = reaction[0], reaction[1]
  reactants = reactants.split('.')

  product_fgs_list = mol_fgs(product)
  reactant_fgs_list = [mol_fgs(r) for r in reactants]
  return (reactant_fgs_list, product_fgs_list)


def worker(reactions_chunk, results_queue, chunk_idx):
    results = []
    for c, reaction in enumerate(reactions_chunk): 
      results.append(rxn_fgs(reaction))
      if c%10000 ==0 : print(f'{chunk_idx}: {c/len(reactions_chunk) * 100} done')

    results_queue.put([chunk_idx ,results])

def chunk_list(lst, n):
    """
    Divide a list into n chunks. If the size of the list is not divisible
    by n, the last chunk will be of variable length.
    """
    quotient = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    
    for i in range(n):
        if i < remainder: end = start + quotient + 1
        else: end = start + quotient
        result.append(lst[start:end])
        start = end 
    return result

def process_reactions(reactions, num_processes):
    reactions_chunks = chunk_list(reactions, num_processes)
    results_queue = mp.Queue()
    processes = [
        mp.Process(target=worker, args=(reactions_chunk, results_queue, chunk_idx))
        for chunk_idx, reactions_chunk in enumerate(reactions_chunks)
    ]

    for process in processes: process.start()
    results = []
    for _ in range(num_processes): results.append(results_queue.get())
    for process in processes: process.join()
    

    return results


def build_fg_vocab(reactions):
  fgvocab = FGVocab()

  for reactants, product in reactions:
    
    for reactant in reactants:
      for fg, count in reactant: fgvocab.add_to_vocab(fg)
        
    for fg, count in product: fgvocab.add_to_vocab(fg)

  return fgvocab

def idx_tokenize_reactions(reactions, fgvocab):
  fg_reaction_vectors = []

  for reaction in reactions:
    reactant_vectors = fgvocab.vectorize_reaction(reaction)
    fg_reaction_vectors.append(reactant_vectors)

  return fg_reaction_vectors



if __name__ == '__main__':
    df_test = pd.read_csv('dataset/ocrtest.csv', names=['reactants', 'products'])
    df_train = pd.read_csv('dataset/ocrtrain.csv', names=['reactants', 'products'])

    frac = 0.4

    train = df_train.sample(frac=frac, random_state = 1)
    val = train.sample(frac=0.1, random_state = 1)
    train = train.drop(val.index)
    train_data = list(zip(train.reactants.tolist(), train.products.tolist()))
    val_data = list(zip(val.reactants.tolist(), val.products.tolist()))
    test = df_test.sample(frac=frac)
    test_data = list(zip(test.reactants.tolist(), test.products.tolist()))

    all_conversations = train_data
    eval_conversations = val_data
    test_conversations = test_data
    all_fg = process_reactions(all_conversations, mp.cpu_count())
    val_fg = process_reactions(eval_conversations, mp.cpu_count())
    test_fg = process_reactions(eval_conversations, mp.cpu_count())

    print("Finished annotating reactions with fgs")
    all_fg = [reaction for reaction_list in sorted(all_fg, key = lambda x : x[0]) for reaction in reaction_list[1]]
    val_fg = [reaction for reaction_list in sorted(val_fg, key = lambda x : x[0]) for reaction in reaction_list[1]]
    test_fg = [reaction for reaction_list in sorted(test_fg, key = lambda x : x[0]) for reaction in reaction_list[1]]
    print("Finished sorting reactions with fgs")
    fg_vocab = build_fg_vocab(all_fg)
    print("Finished building vocab for fgs")
    all_fg = idx_tokenize_reactions(all_fg, fg_vocab)
    val_fg = idx_tokenize_reactions(val_fg, fg_vocab)
    test_fg = idx_tokenize_reactions(test_fg, fg_vocab)
    print("Finished tokenizing reactions")

    data = {
        'all_conversations ': all_conversations,
        'eval_conversations' : eval_conversations,
        'test_conversations':test_conversations,
        'fg_vocab': fg_vocab,
        'all_conversations_fg': all_fg,
        'eval_conversations_fg': val_fg,
        'test_conversations_fg': test_fg
    }
    print("creating pickle")

    with open('start_data.pkl', 'wb') as file: pickle.dump(data, file)

    print("finished")
