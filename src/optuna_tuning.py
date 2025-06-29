#!/usr/bin/env python
import os
import sys
import yaml
import optuna
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our modules
from dataset import (MRIDataset, WSIDataset, MultiModalDataset, 
                    custom_collate_fn_mri, custom_collate_fn_wsi, custom_collate_fn_multimodal)
from model import MRIEDLModel, WSIEDLModel, MultiModalEvidentialModel
from losses import CombinedLoss, MultiModalCombinedLoss
from training import run_training, evaluate_mri, evaluate_wsi, evaluate_multimodal


class OptunaMultiObjectiveTrainer:
    def __init__(self, base_config_path='config.yaml', modality='multi', n_trials=100, study_name='medical_ai_multi_optimization'):
        self.base_config_path = base_config_path
        self.modality = modality
        self.n_trials = n_trials
        self.study_name = study_name
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Load and prepare data once
        self._prepare_data()
        
        # Create MULTI-OBJECTIVE study
        self.study = optuna.create_study(
            directions=['maximize', 'maximize'],  # Maximize both C-index and AUC
            study_name=study_name,
            storage=f'sqlite:///{study_name}.db',
            load_if_exists=True
        )
        
        print(f"Multi-Objective Optuna Optimization for {modality.upper()} modality")
        print(f"Objectives: C-index ↑ AND AUC ↑ (independent optimization)")
        print(f"Dataset: {len(self.clinical_df)} patients")
        print(f"Running {n_trials} trials")
        print("="*80)
    
    def _prepare_data(self):
        """Load and prepare data once"""
        self.clinical_df = pd.read_csv(self.base_config['data']['clinical_file'])
        patient_ids = self.clinical_df['patient_id'].values
        labels = self.clinical_df['BCR'].values
        
        # Train-validation-test split for proper optimization
        train_val_ids, self.test_ids = train_test_split(
            patient_ids, test_size=0.2, stratify=labels, random_state=42
        )
        train_val_labels = labels[np.isin(patient_ids, train_val_ids)]
        
        self.train_ids, self.val_ids = train_test_split(
            train_val_ids, test_size=0.25, stratify=train_val_labels, random_state=42
        )
        
        print(f"Data split: Train={len(self.train_ids)}, Val={len(self.val_ids)}, Test={len(self.test_ids)}")
    
    def suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for optimization"""
        config = self.base_config.copy()
        
        # Model architecture parameters
        config['model']['d_model'] = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        config['model']['n_layers'] = trial.suggest_int('n_layers', 1, 5)
        config['model']['d_state'] = trial.suggest_categorical('d_state', [8, 16, 32, 64])
        config['model']['d_conv'] = trial.suggest_int('d_conv', 2, 8)
        config['model']['expand'] = trial.suggest_categorical('expand', [1, 2, 4])
        config['model']['n_heads'] = trial.suggest_categorical('n_heads', [2, 4, 8, 16])
        config['model']['dropout'] = trial.suggest_float('dropout', 0.1, 0.7)
        
        if self.modality in ['mri', 'multi']:
            config['model']['max_slices_attention'] = trial.suggest_categorical('max_slices_attention', [500, 1000, 2000, 3000])
        if self.modality in ['wsi', 'multi']:
            config['model']['max_patches_attention'] = trial.suggest_categorical('max_patches_attention', [2000, 5000, 10000, 15000])
        
        # Training parameters
        config['training']['batch_size'] = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
        config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        config['training']['gradient_clip'] = trial.suggest_float('gradient_clip', 0.1, 2.0)
        
        # Loss function parameters
        if self.modality in ['mri', 'wsi']:
            config['loss']['cox_weight'] = trial.suggest_float('cox_weight', 0.5, 3.0)
            config['loss']['edl_weight'] = trial.suggest_float('edl_weight', 0.1, 1.0)
        elif self.modality == 'multi':
            config['loss']['cox_weight'] = trial.suggest_float('cox_weight', 0.5, 2.0)
            config['loss']['edl_weight'] = trial.suggest_float('edl_weight', 0.1, 1.0)
            config['loss']['focal_weight'] = trial.suggest_float('focal_weight', 0.1, 1.0)
        
        config['loss']['lamb'] = trial.suggest_float('lamb', 1e-5, 1e-2, log=True)
        
        # Scheduler parameters
        config['scheduler']['factor'] = trial.suggest_float('scheduler_factor', 0.3, 0.9)
        config['scheduler']['patience'] = trial.suggest_int('scheduler_patience', 2, 8)
        
        # Data parameters
        if self.modality in ['mri', 'multi']:
            config['data']['max_mri_slices'] = trial.suggest_categorical('max_mri_slices', [20, 50, 100, 200])
        if self.modality in ['wsi', 'multi']:
            config['data']['max_wsi_patches'] = trial.suggest_categorical('max_wsi_patches', [5000, 10000, 20000, 30000])
        
        # Reduce epochs for faster optimization
        config['training']['epochs'] = 25
        config['training']['early_stopping_patience'] = 5
        
        return config
    
    def create_datasets_and_loaders(self, config):
        """Create datasets and loaders for optimization"""
        modality = config['modality']
        batch_size = config['training']['batch_size']
        
        if modality == 'mri':
            train_dataset = MRIDataset(
                mri_data_dir=config['data']['mri_data_dir'],
                clinical_df=self.clinical_df,
                patient_ids=self.train_ids,
                clinical_features=config['clinical_features'],
                max_slices=config['data']['max_mri_slices']
            )
            
            val_dataset = MRIDataset(
                mri_data_dir=config['data']['mri_data_dir'],
                clinical_df=self.clinical_df,
                patient_ids=self.val_ids,
                clinical_features=config['clinical_features'],
                clinical_scaler=train_dataset.clinical_scaler,
                mri_scaler=train_dataset.mri_scaler,
                max_slices=config['data']['max_mri_slices']
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_mri)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_mri)
            
        elif modality == 'wsi':
            train_dataset = WSIDataset(
                wsi_features_dir=config['data']['wsi_features_dir'],
                clinical_df=self.clinical_df,
                patient_ids=self.train_ids,
                clinical_features=config['clinical_features'],
                max_patches=config['data']['max_wsi_patches']
            )
            
            val_dataset = WSIDataset(
                wsi_features_dir=config['data']['wsi_features_dir'],
                clinical_df=self.clinical_df,
                patient_ids=self.val_ids,
                clinical_features=config['clinical_features'],
                clinical_scaler=train_dataset.clinical_scaler,
                max_patches=config['data']['max_wsi_patches']
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_wsi)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_wsi)
            
        elif modality == 'multi':
            train_dataset = MultiModalDataset(
                mri_data_dir=config['data']['mri_data_dir'],
                wsi_features_dir=config['data']['wsi_features_dir'],
                clinical_df=self.clinical_df,
                patient_ids=self.train_ids,
                clinical_features=config['clinical_features'],
                max_mri_slices=config['data']['max_mri_slices'],
                max_wsi_patches=config['data']['max_wsi_patches']
            )
            
            val_dataset = MultiModalDataset(
                mri_data_dir=config['data']['mri_data_dir'],
                wsi_features_dir=config['data']['wsi_features_dir'],
                clinical_df=self.clinical_df,
                patient_ids=self.val_ids,
                clinical_features=config['clinical_features'],
                clinical_scaler=train_dataset.clinical_scaler,
                mri_scaler=train_dataset.mri_scaler,
                max_mri_slices=config['data']['max_mri_slices'],
                max_wsi_patches=config['data']['max_wsi_patches']
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_multimodal)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_multimodal)
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def create_model(self, config, train_dataset, device):
        """Create model for optimization"""
        modality = config['modality']
        
        if modality == 'mri':
            model = MRIEDLModel(
                mri_dim=train_dataset.n_mri_features,
                clinical_dim=train_dataset.clinical_data.shape[1],
                config=config
            ).to(device)
            
        elif modality == 'wsi':
            model = WSIEDLModel(
                wsi_dim=1024,
                clinical_dim=train_dataset.clinical_data.shape[1],
                config=config
            ).to(device)
            
        elif modality == 'multi':
            model = MultiModalEvidentialModel(
                mri_dim=train_dataset.n_mri_features,
                wsi_dim=1024,
                clinical_dim=train_dataset.clinical_data.shape[1],
                config=config
            ).to(device)
        
        return model
    
    def create_loss_function(self, config):
        """Create loss function for optimization"""
        modality = config['modality']
        
        if modality in ['mri', 'wsi']:
            criterion = CombinedLoss(
                cox_weight=config['loss']['cox_weight'],
                edl_weight=config['loss']['edl_weight'],
                lamb=config['loss']['lamb']
            )
        elif modality == 'multi':
            criterion = MultiModalCombinedLoss(
                cox_weight=config['loss']['cox_weight'],
                edl_weight=config['loss']['edl_weight'],
                focal_weight=config['loss']['focal_weight'],
                lamb=config['loss']['lamb']
            )
        
        return criterion
    
    def objective(self, trial):
        """Multi-objective function - returns [cindex, auc]"""
        try:
            # Suggest hyperparameters
            config = self.suggest_hyperparameters(trial)
            
            # Save temporary config
            temp_config_path = f'temp_config_trial_{trial.number}.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Setup device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create datasets and loaders
            train_dataset, val_dataset, train_loader, val_loader = self.create_datasets_and_loaders(config)
            
            # Create model
            model = self.create_model(config, train_dataset, device)
            
            # Create loss function
            criterion = self.create_loss_function(config)
            
            # Create optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config['training']['learning_rate'], 
                weight_decay=config['training']['weight_decay']
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=config['scheduler']['mode'], 
                factor=config['scheduler']['factor'], 
                patience=config['scheduler']['patience'], 
                verbose=False, 
                min_lr=config['scheduler']['min_lr']
            )
            
            # Disable saving for optimization
            config['output']['save_model'] = False
            
            # Run training (with minimal output)
            history, best_metric, best_epoch = run_training(
                model, train_loader, val_loader, optimizer, criterion, 
                scheduler, config, device, self.modality
            )
            
            # Get final validation results
            if self.modality == 'mri':
                final_results = evaluate_mri(model, val_loader, device)
                cindex = final_results['cindex']
                auc = final_results['auc']
            elif self.modality == 'wsi':
                final_results = evaluate_wsi(model, val_loader, device)
                cindex = final_results['cindex']
                auc = final_results['auc']
            elif self.modality == 'multi':
                final_results = evaluate_multimodal(model, val_loader, device)
                cindex = final_results['fused_cindex']
                auc = final_results['fused_auc']
            
            # Store additional metrics
            trial.set_user_attr('cindex', cindex)
            trial.set_user_attr('auc', auc)
            trial.set_user_attr('best_epoch', best_epoch)
            
            # Clean up
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
            
            # Return BOTH objectives independently
            return cindex, auc
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            # Clean up temp config on failure
            temp_config_path = f'temp_config_trial_{trial.number}.yaml'
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            # Return very low scores for failed trials
            return 0.0, 0.0
    
    def run_optimization(self):
        """Run the multi-objective optimization"""
        print(f"Starting Multi-Objective Optuna optimization...")
        print(f"Objective 1: Maximize C-index")
        print(f"Objective 2: Maximize AUC")
        print(f"Finding Pareto frontier...")
        
        # Add custom callbacks for progress tracking
        def print_progress(study, trial):
            if trial.number % 5 == 0:  # Print every 5 trials
                cindex = trial.user_attrs.get('cindex', 0)
                auc = trial.user_attrs.get('auc', 0)
                print(f"Trial {trial.number:3d}: C-index={cindex:.4f} | AUC={auc:.4f}")
        
        # Run optimization
        self.study.optimize(
            self.objective, 
            n_trials=self.n_trials,
            callbacks=[print_progress],
            show_progress_bar=True
        )
        
        print(f"\n Multi-objective optimization completed!")
        return self.study
    
    def get_pareto_front_configs(self, n_configs=5):
        """Get multiple configs from the Pareto front"""
        # Get Pareto optimal trials
        pareto_trials = self.study.best_trials
        
        configs = []
        for i, trial in enumerate(pareto_trials[:n_configs]):
            config = self.base_config.copy()
            
            # Update with trial parameters
            for key, value in trial.params.items():
                if key in ['d_model', 'n_layers', 'd_state', 'd_conv', 'expand', 'n_heads', 'dropout', 'max_slices_attention', 'max_patches_attention']:
                    config['model'][key] = value
                elif key in ['batch_size', 'learning_rate', 'weight_decay', 'gradient_clip']:
                    config['training'][key] = value
                elif key in ['cox_weight', 'edl_weight', 'focal_weight', 'lamb']:
                    config['loss'][key] = value
                elif key.startswith('scheduler_'):
                    param_name = key.replace('scheduler_', '')
                    config['scheduler'][param_name] = value
                elif key in ['max_mri_slices', 'max_wsi_patches']:
                    config['data'][key] = value
            
            # Restore full epochs for final training
            config['training']['epochs'] = 50
            config['training']['early_stopping_patience'] = 8
            
            cindex = trial.user_attrs.get('cindex', 0)
            auc = trial.user_attrs.get('auc', 0)
            
            configs.append({
                'config': config,
                'trial_number': trial.number,
                'cindex': cindex,
                'auc': auc,
                'values': trial.values
            })
        
        return configs
    
    def save_pareto_configs(self, output_dir='pareto_configs'):
        """Save multiple configs from Pareto front"""
        pareto_configs = self.get_pareto_front_configs(n_configs=5)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each config
        for i, config_data in enumerate(pareto_configs):
            filename = f"{output_dir}/config_pareto_{i+1}_trial_{config_data['trial_number']}.yaml"
            with open(filename, 'w') as f:
                yaml.dump(config_data['config'], f, default_flow_style=False)
            
            print(f" Config {i+1}: {filename}")
            print(f"   C-index: {config_data['cindex']:.4f} | AUC: {config_data['auc']:.4f}")
        
        # Save best C-index config
        best_cindex_config = max(pareto_configs, key=lambda x: x['cindex'])
        with open('config_best_cindex.yaml', 'w') as f:
            yaml.dump(best_cindex_config['config'], f, default_flow_style=False)
        print(f"\n Best C-index config: config_best_cindex.yaml (C-index: {best_cindex_config['cindex']:.4f})")
        
        # Save best AUC config
        best_auc_config = max(pareto_configs, key=lambda x: x['auc'])
        with open('config_best_auc.yaml', 'w') as f:
            yaml.dump(best_auc_config['config'], f, default_flow_style=False)
        print(f" Best AUC config: config_best_auc.yaml (AUC: {best_auc_config['auc']:.4f})")
        
        return pareto_configs
    
    def print_study_summary(self):
        """Print multi-objective study summary"""
        print(f"\n" + "="*80)
        print(f" MULTI-OBJECTIVE OPTUNA OPTIMIZATION SUMMARY")
        print(f"="*80)
        print(f"Study name: {self.study.study_name}")
        print(f"Number of trials: {len(self.study.trials)}")
        print(f"Number of Pareto optimal solutions: {len(self.study.best_trials)}")
        
        # Print Pareto front
        print(f"\n PARETO OPTIMAL SOLUTIONS (best trade-offs):")
        for i, trial in enumerate(self.study.best_trials[:10]):  # Top 10
            cindex = trial.user_attrs.get('cindex', trial.values[0] if trial.values else 0)
            auc = trial.user_attrs.get('auc', trial.values[1] if trial.values and len(trial.values) > 1 else 0)
            print(f"  {i+1:2d}. Trial {trial.number:3d}: C-index={cindex:.4f} | AUC={auc:.4f}")
        
        # Find extreme points
        if self.study.best_trials:
            best_cindex_trial = max(self.study.best_trials, key=lambda t: t.user_attrs.get('cindex', 0))
            best_auc_trial = max(self.study.best_trials, key=lambda t: t.user_attrs.get('auc', 0))
            
            print(f"\n EXTREME POINTS:")
            print(f"   Best C-index: Trial {best_cindex_trial.number} | C-index={best_cindex_trial.user_attrs.get('cindex', 0):.4f} | AUC={best_cindex_trial.user_attrs.get('auc', 0):.4f}")
            print(f"   Best AUC:     Trial {best_auc_trial.number} | C-index={best_auc_trial.user_attrs.get('cindex', 0):.4f} | AUC={best_auc_trial.user_attrs.get('auc', 0):.4f}")
        
        print(f"="*80)


def main():
    """Main function for Multi-Objective Optuna optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Objective Optuna Hyperparameter Optimization')
    parser.add_argument('--modality', choices=['mri', 'wsi', 'multi'], default='multi',
                        help='Modality to optimize')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--config', default='config.yaml',
                        help='Base configuration file')
    parser.add_argument('--study_name', default=None,
                        help='Optuna study name')
    
    args = parser.parse_args()
    
    if args.study_name is None:
        args.study_name = f'{args.modality}_multi_objective_optimization'
    
    # Create trainer
    trainer = OptunaMultiObjectiveTrainer(
        base_config_path=args.config,
        modality=args.modality,
        n_trials=args.n_trials,
        study_name=args.study_name
    )
    
    # Run optimization
    study = trainer.run_optimization()
    
    # Save Pareto optimal configs
    pareto_configs = trainer.save_pareto_configs()
    
    # Print summary
    trainer.print_study_summary()
    
    return study, pareto_configs


if __name__ == "__main__":
    study, pareto_configs = main()