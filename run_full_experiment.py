#!/usr/bin/env python3
"""
Full experiment runner for Module 11 (AI Privacy) — runs full target + shadow + attack pipeline.

Notes:
- This script depends on `htb_ai_library` described in the module notes.
- It saves models and figures under `11-AI-Privacy/output/` and `11-AI-Privacy/Images/`.
"""
import os
import time
import logging

import torch

from htb_ai_library import (
    set_reproducibility, use_htb_style,
    MLP, AttackModel,
    load_adult_census,
    train_fixed_epochs, train_with_early_stopping, evaluate_model,
    get_model_predictions, prepare_attack_data, create_dataloader,
    plot_training_history, plot_overfitting_gap, plot_confidence_distributions,
    plot_shadow_confidence_distributions, plot_attack_roc_curve, plot_precision_recall_curve,
    plot_attack_accuracy_comparison, analyze_attack_decision_boundary, plot_decision_boundary,
)


LOGDIR = os.path.join('11-AI-Privacy', 'output')
IMAGEDIR = os.path.join('11-AI-Privacy', 'Images')
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(IMAGEDIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('full_run')


def main():
    RANDOM_SEED = 1337
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_reproducibility(RANDOM_SEED)
    use_htb_style()

    OUTPUT_DIR = LOGDIR
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    FIGS_DIR = IMAGEDIR
    os.makedirs(MODEL_DIR, exist_ok=True)

    DATASET_CONFIG = {"num_classes": 2}

    TARGET_MODEL_CONFIG = {
        "hidden_layers": [256, 128],
        "dropout": 0.0,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
    }

    SHADOW_MODEL_CONFIG = {
        "num_shadow_models": 5,
        "hidden_layers": [128, 64],
        "dropout": 0.3,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "early_stopping_patience": 10,
        "shadow_data_size": 0.5,
    }

    ATTACK_MODEL_CONFIG = {
        "hidden_layers": [64, 32],
        "dropout": 0.2,
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.001,
        "early_stopping_patience": 15,
    }

    logger.info('Loading dataset')
    X_target, y_target, X_shadow, y_shadow, X_attack_eval, y_attack_eval, num_features = load_adult_census(random_state=RANDOM_SEED)
    logger.info(f'Features: {num_features}, target samples: {len(X_target)}, shadow: {len(X_shadow)}, attack eval: {len(X_attack_eval)}')

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_target_norm = scaler.fit_transform(X_target)
    X_attack_eval_norm = scaler.transform(X_attack_eval)

    # DataLoaders
    train_loader = create_dataloader(X_target_norm, y_target, TARGET_MODEL_CONFIG['batch_size'])
    test_loader = create_dataloader(X_attack_eval_norm, y_attack_eval, TARGET_MODEL_CONFIG['batch_size'], shuffle=False)

    # Target model
    target_model = MLP(input_size=num_features, hidden_layers=TARGET_MODEL_CONFIG['hidden_layers'], num_classes=DATASET_CONFIG['num_classes'], dropout=TARGET_MODEL_CONFIG['dropout'])
    target_model.to(DEVICE)
    logger.info('Training target model (this may take a long time)')
    history = train_fixed_epochs(target_model, train_loader, test_loader, device=DEVICE, epochs=TARGET_MODEL_CONFIG['epochs'], learning_rate=TARGET_MODEL_CONFIG['learning_rate'])
    plot_training_history(history, save_path=os.path.join(FIGS_DIR, '335_Introduction_target_training.png'))

    train_acc, _, _ = evaluate_model(target_model, train_loader, DEVICE)
    test_acc, _, _ = evaluate_model(target_model, test_loader, DEVICE)
    logger.info(f'Target Training Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, gap: {train_acc - test_acc:.4f}')
    plot_overfitting_gap(train_acc, test_acc, save_path=os.path.join(FIGS_DIR, '335_Introduction_overfitting_gap.png'))

    # Shadow models + prepare attack dataset
    logger.info('Training shadow models and preparing attack dataset (this will also take time)')
    member_preds = []
    nonmember_preds = []
    member_labels = []
    nonmember_labels = []

    # For reproducibility: split shadow data into num_shadow_models parts
    # Here we use a simple repeated training on random subsets
    import numpy as np
    n_shadow = SHADOW_MODEL_CONFIG['num_shadow_models']
    total_shadow = len(X_shadow)
    idx = np.arange(total_shadow)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(idx)
    split_size = total_shadow // n_shadow

    for i in range(n_shadow):
        s = i * split_size
        e = s + split_size if i < n_shadow - 1 else total_shadow
        Xi = X_shadow[idx[s:e]]
        yi = y_shadow[idx[s:e]]

        # build dataloaders
        Xi_norm = scaler.transform(Xi)
        shadow_loader = create_dataloader(Xi_norm, yi, SHADOW_MODEL_CONFIG['batch_size'])

        shadow_model = MLP(input_size=num_features, hidden_layers=SHADOW_MODEL_CONFIG['hidden_layers'], num_classes=DATASET_CONFIG['num_classes'], dropout=SHADOW_MODEL_CONFIG['dropout'])
        shadow_model.to(DEVICE)
        train_with_early_stopping(shadow_model, shadow_loader, device=DEVICE, epochs=SHADOW_MODEL_CONFIG['epochs'], learning_rate=SHADOW_MODEL_CONFIG['learning_rate'], patience=SHADOW_MODEL_CONFIG['early_stopping_patience'])

        # collect member preds
        mpreds = get_model_predictions(shadow_model, Xi_norm, device=DEVICE)
        member_preds.append(mpreds)
        member_labels.append(yi)

        # produce non-member predictions using held-out attack eval split of same size
        # sample random non-members from attack eval
        import numpy.random as npr
        pick = npr.choice(len(X_attack_eval), size=len(Xi), replace=False)
        Xnm = X_attack_eval[pick]
        ynm = y_attack_eval[pick]
        Xnm_norm = scaler.transform(Xnm)
        nmpreds = get_model_predictions(shadow_model, Xnm_norm, device=DEVICE)
        nonmember_preds.append(nmpreds)
        nonmember_labels.append(ynm)

    # prepare attack data
    X_attack_train, y_attack_train = prepare_attack_data(member_preds, nonmember_preds, member_labels, nonmember_labels)

    # train attack model
    logger.info('Training attack model')
    attack_loader = create_dataloader(X_attack_train, y_attack_train, ATTACK_MODEL_CONFIG['batch_size'])
    attack_model = AttackModel(input_size=X_attack_train.shape[1], hidden_layers=ATTACK_MODEL_CONFIG['hidden_layers'], dropout=ATTACK_MODEL_CONFIG['dropout'])
    attack_model.to(DEVICE)
    train_with_early_stopping(attack_model, attack_loader, device=DEVICE, epochs=ATTACK_MODEL_CONFIG['epochs'], learning_rate=ATTACK_MODEL_CONFIG['learning_rate'], patience=ATTACK_MODEL_CONFIG['early_stopping_patience'])

    # evaluate attack against real target predictions
    logger.info('Evaluating attack on target model')
    target_member_preds = get_model_predictions(target_model, X_target_norm, device=DEVICE)
    target_nonmember_preds = get_model_predictions(target_model, X_attack_eval_norm, device=DEVICE)
    X_attack_test, y_attack_test = prepare_attack_data([target_member_preds], [target_nonmember_preds], [y_target], [y_attack_eval])

    # evaluate attack model
    test_loader_attack = create_dataloader(X_attack_test, y_attack_test, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)
    atk_acc, atk_preds, atk_probs = evaluate_model(attack_model, test_loader_attack, DEVICE)
    logger.info(f'Attack accuracy: {atk_acc:.4f}')

    # save figures summarizing attack
    plot_attack_roc_curve(y_attack_test, atk_probs, save_path=os.path.join(FIGS_DIR, 'attack_roc.png'))
    plot_precision_recall_curve(y_attack_test, atk_probs, save_path=os.path.join(FIGS_DIR, 'attack_pr.png'))
    plot_attack_accuracy_comparison(atk_acc, save_path=os.path.join(FIGS_DIR, 'attack_accuracy.png'))

    logger.info('Full run complete — figures saved in %s', FIGS_DIR)


if __name__ == '__main__':
    main()
