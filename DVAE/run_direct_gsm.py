import torch
from discrete_vae import DVAE
from discrete_vae import GSM
params = {'num_epochs': 300,
            'composed_decoder': True,
            'batch_size': 100,
            'learning_rate': 0.001,
            'gumbels' : 1,
            'N_K': (10,2),
            'eps_0':1.0,
            'anneal_rate':1e-5,
            'min_eps':0.1,
            'random_seed':777,
            'dataset':'mnist', # 'mnist' or 'omniglot'
            'split_valid':True,
            'binarize':True,
            'ST-estimator':False, # relevant only for GSM
            'save_images':False,
            'print_result':True}




"""
returned results:

 train_results: list, where the i'th element is the average nll of the mini-batches of i'th epoch,
 test_results: list, where the i'th element is the average nll of the mini-batches of i'th epoch,
 best_test_nll: the test nll of the epoch where the validation nll is the best of all epochs,
 best_state_dicts: pytorch models,
 params
 """
dvae_results = DVAE.training_procedure(params)

#gsm_results = GSM.training_procedure(params)

