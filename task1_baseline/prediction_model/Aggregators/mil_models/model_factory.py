import os
from pathlib import Path
from task1_baseline.prediction_model.Aggregators.mil_models import ABMIL, ABMIL_FUSION
from task1_baseline.prediction_model.Aggregators.mil_models import ABMILConfig

import pdb
import torch
from task1_baseline.prediction_model.Aggregators.utils.file_utils import save_pkl, load_pkl
from os.path import join as j_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_downstream_model(args, mode='classification', config_dir=None):
    """
    Create downstream modles for classification or survival
    """
    if config_dir is None:
        # This makes the path robust to where the script is run from.
        # It constructs the path relative to this file's location.
        # .../mil_models/ -> .../Aggregators/ -> .../Aggregators/configs
        base_dir = Path(__file__).resolve().parent.parent
        config_dir = base_dir / 'configs'

    config_path = os.path.join(str(config_dir), args.model_config, 'config.json')
    assert os.path.exists(config_path), f"Config path {config_path} doesn't exist!"
    
    model_config = args.model_config
    model_type = args.model_type

    if 'IndivMLPEmb' in model_config:
        print(args)
        update_dict = {'in_dim': args.in_dim,
                       'p': 16,
                       'out_type': args.out_type,
                       }

    elif model_type == 'DeepAttnMIL':
        update_dict = {'in_dim': args.in_dim,
                       'out_size': args.out_size,
                       'load_proto': args.load_proto,
                       'fix_proto': args.fix_proto,
                       'proto_path': args.proto_path}
    else:
        update_dict = {'in_dim': args.in_dim}

    # Always add fusion_dim for ABMIL models to support fusion capabilities
    if model_type == 'ABMIL':
        update_dict.update({
            'fusion_dim': 320,  # MRI feature dimension
            'clinical_dim': 64,  # Default clinical feature dimension
            'clinical_hidden_dim': 64,  # Default hidden dimension for clinical MLP
            'clinical_layers': 2  # Default number of layers in clinical MLP
        })

    # Set clinical dimension based on the clinical processor if available
    if model_type == 'ABMIL' and hasattr(args, 'clinical_processor') and args.clinical_processor is not None:
        clinical_dim = args.clinical_processor.output_dim
        update_dict.update({'clinical_dim': clinical_dim})
        print(f"Setting clinical_dim to {clinical_dim} based on clinical processor")
    
    if mode == 'classification':
        update_dict.update({'n_classes': args.n_classes})
    elif mode == 'survival':
        if args.loss_fn == 'nll':
            update_dict.update({'n_classes': args.n_label_bins})
        elif args.loss_fn == 'cox':
            update_dict.update({'n_classes': 1})
        elif args.loss_fn == 'rank':
            update_dict.update({'n_classes': 1})
    else:
        raise NotImplementedError(f"Not implemented for {mode}...")
    
    if model_type == 'ABMIL':
        config = ABMILConfig.from_pretrained(config_path, update_dict=update_dict)
        print(f"Model factory: Config loaded with fusion_dim = {getattr(config, 'fusion_dim', 'NOT FOUND')}")
        print(f"Model factory: Update dict = {update_dict}")
        model = ABMIL_FUSION(config=config, mode=mode)  # Use fusion model by default
    # Prototype-based models will choose from the following
    elif model_type == 'LinearEmb':
        config = LinearEmbConfig.from_pretrained(config_path, update_dict=update_dict)
        model = LinearEmb(config=config, mode=mode)
    elif 'IndivMLPEmb' in model_type:            
        if 'IndivMLPEmb_Shared' == model_type:
            config = IndivMLPEmbConfig_Shared.from_pretrained(config_path, update_dict=update_dict)
        elif 'IndivMLPEmb_Indiv' == model_type:
            config = IndivMLPEmbConfig_Indiv.from_pretrained(config_path, update_dict=update_dict)
        elif 'IndivMLPEmb_SharedPost' == model_type:
            config = IndivMLPEmbConfig_SharedPost.from_pretrained(config_path, update_dict=update_dict)
        elif 'IndivMLPEmb_IndivPost' == model_type:
            config = IndivMLPEmbConfig_IndivPost.from_pretrained(config_path, update_dict=update_dict)
        elif 'IndivMLPEmb_SharedIndiv' == model_type:
            config = IndivMLPEmbConfig_SharedIndiv.from_pretrained(config_path, update_dict=update_dict)
        elif 'IndivMLPEmb_SharedIndivPost' == model_type:
            config = IndivMLPEmbConfig_SharedIndivPost.from_pretrained(config_path, update_dict=update_dict)
        print(mode)
        model = IndivMLPEmb(config=config, mode=mode)
    else:
        raise NotImplementedError

    return model


def prepare_emb(datasets, args, mode='classification'):
    """
    Slide representation construction with patch feature aggregation trained in unsupervised manner
    """
   
    ### Preparing file path for saving embeddings
    print('Constructing unsupervised slide embedding...', end=' ')
    embeddings_kwargs = {
        'feats': args.data_source[0].split('/')[-2],
        'model_type': args.model_type,
        'out_size': args.n_proto
    }

    # Create embedding path
    fpath = "{feats}_{model_type}_embeddings_proto_{out_size}".format(**embeddings_kwargs)
    if args.model_type == 'PANTHER':
        DIEM_kwargs = {'tau': args.tau, 'out_type': args.out_type, 'eps': args.ot_eps, 'em_step': args.em_iter}
        name = '_{out_type}_em_{em_step}_eps_{eps}_tau_{tau}'.format(**DIEM_kwargs)
        fpath += name
    elif args.model_type == 'OT':
        OTK_kwargs = {'out_type': args.out_type, 'eps': args.ot_eps}
        name = '_{out_type}_eps_{eps}'.format(**OTK_kwargs)
        fpath += name
    embeddings_fpath = j_(args.split_dir, 'embeddings', fpath+'.pkl')
    
    ### Load existing embeddings if already created
    if os.path.isfile(embeddings_fpath):
        embeddings = load_pkl(embeddings_fpath)
        for k, loader in datasets.items():
            print(f'Embedding already exists! Loading {k}', end=' ')
            loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y']
    else:
        os.makedirs(j_(args.split_dir, 'embeddings'), exist_ok=True)
        
        model = create_embedding_model(args, mode=mode).to(device)

        ### Extracts prototypical features per split
        embeddings = {}
        for split, loader in datasets.items():
            print(f"Aggregating {split} set features...")
            X, y = model.predict(loader,
                                 use_cuda=torch.cuda.is_available()
                                 )
            loader.dataset.X, loader.dataset.y = X, y
            embeddings[split] = {'X': X, 'y': y}
        save_pkl(embeddings_fpath, embeddings)

    return datasets, embeddings_fpath