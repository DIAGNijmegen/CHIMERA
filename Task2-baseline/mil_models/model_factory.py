import os
from mil_models.model_abmil_fusion import  ABMIL_Fusion, ABMIL_Fusion_BN

from mil_models.tabular_snn import TabularSNN  # If needed elsewhere
# If ABMIL_Fusion is not in mil_models, import it where it's defined
# from your_custom_module import ABMIL_Fusion

from os.path import join as j_

import torch
from utils.file_utils import save_pkl, load_pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_downstream_model(args, mode='classification', config_dir=None):
    """
    Create ABMIL or ABMIL_Fusion model for classification
    """
    # Default config directory inside ABMIL_CHIMERA/configs/
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
        config_dir = os.path.abspath(config_dir)

    update_dict = {'in_dim': args.in_dim}
    if mode == 'classification':
        update_dict.update({'n_classes': args.n_classes})
    else:
        raise NotImplementedError("Only classification is supported")

   
    if args.model_type == 'ABMIL_Fusion':
        model = ABMIL_Fusion(
            in_dim=args.in_dim,
            clinical_in_dim=args.clinical_in_dim,
            n_classes=args.n_classes
        )
    else:
        config_path = os.path.join(config_dir, args.model_config, 'config.json')
        assert os.path.exists(config_path), f"Config path {config_path} doesn't exist!"
        config = ABMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = ABMIL(config=config, mode=mode)

    return model


def prepare_emb(datasets, args, mode='classification'):
    """
    Slide representation construction with patch feature aggregation trained in unsupervised manner
    """
    print('\nConstructing unsupervised slide embedding...', end=' ')
    embeddings_kwargs = {
        'feats': args.data_source[0].split('/')[-2],
        'model_type': args.model_type,
        'out_size': args.n_proto
    }

    fpath = "{feats}_{model_type}_embeddings_proto_{out_size}".format(**embeddings_kwargs)
    if args.model_type == 'PANTHER':
        DIEM_kwargs = {'tau': args.tau, 'out_type': args.out_type, 'eps': args.ot_eps, 'em_step': args.em_iter}
        name = '_{out_type}_em_{em_step}_eps_{eps}_tau_{tau}'.format(**DIEM_kwargs)
        fpath += name
    elif args.model_type == 'OT':
        OTK_kwargs = {'out_type': args.out_type, 'eps': args.ot_eps}
        name = '_{out_type}_eps_{eps}'.format(**OTK_kwargs)
        fpath += name

    embeddings_fpath = j_(args.split_dir, 'embeddings', fpath + '.pkl')

    # Load existing
    if os.path.isfile(embeddings_fpath):
        embeddings = load_pkl(embeddings_fpath)
        for k, loader in datasets.items():
            print(f'\n\tEmbedding already exists! Loading {k}', end=' ')
            loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y']
    else:
        os.makedirs(j_(args.split_dir, 'embeddings'), exist_ok=True)

        model = create_embedding_model(args, mode=mode).to(device)

        embeddings = {}
        for split, loader in datasets.items():
            print(f"\nAggregating {split} set features...")
            X, y = model.predict(loader, use_cuda=torch.cuda.is_available())
            loader.dataset.X, loader.dataset.y = X, y
            embeddings[split] = {'X': X, 'y': y}
        save_pkl(embeddings_fpath, embeddings)

    return datasets, embeddings_fpath
