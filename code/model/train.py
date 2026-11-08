import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train, validate, test
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

def train_model(model, train_data_loader, val_data_loader, optimizer, criterion, scheduler, config, save_dir):
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    best_acc = 0
    for epoch in range(config['epochs']):
        train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss}\tLR: {curr_lr}')
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss}')
        print(f'Validation Metrics - Accuracy: {val_acc:.2f}% | Precision: {val_prec:.2f}% | Recall: {val_rec:.2f}% | F1: {val_f1:.2f}%\n')
        
        if scheduler is not None:
            if config['sch']['name'] == 'lronplateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss, 
                'val_loss': val_loss, 
                'val_accuracy': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_f1': val_f1,
                'lr': curr_lr
            })

        if val_acc >= best_acc and not config['debug']:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'acc': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1': val_f1
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')
    
    if not config['debug']:
        wandb.finish()
    
    return best_acc

def main():
    # Load config
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    config['device'] = device

    # Load data
    train_data_loader, val_data_loader, test_data_loader = load_data(config)
    config['sch']['steps'] = len(train_data_loader)
    
    # Initialize best test metrics
    best_test_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'run': 0
    }
    
    # Create a file to store all runs
    with open("all_runs_summary.txt", "w") as all_runs_file:
        all_runs_file.write(f"# All Training Runs Summary\n")
        all_runs_file.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        all_runs_file.write(f"# Configuration\n")
        all_runs_file.write(f"- Learning Rate: {config['optim']['lr']}\n")
        all_runs_file.write(f"- Batch Size: {config['batch_size']}\n")
        all_runs_file.write(f"- Epochs: {config['epochs']}\n")
        all_runs_file.write(f"- Scheduler: {config['sch']['name']}\n")
        if config['sch']['name'] == 'lronplateau':
            all_runs_file.write(f"- Scheduler Factor: {config['sch']['factor']}\n")
            all_runs_file.write(f"- Scheduler Patience: {config['sch']['patience']}\n\n")
        all_runs_file.write("# Results Summary\n\n")
    
    # Run training 1 time
    for run in range(1):
        print(f'\nStarting training run {run + 1}/1')
        
        # Create model and training components
        model = create_model(config)
        criterion, optimizer, scheduler = cri_opt_sch(config, model)
        model = model.to(device)
        
        # Initialize wandb and create save directory
        if not config['debug']:
            run_name = f'{config["task"]}-{datetime.now().strftime("%m%d_%H%M")}-run{run+1}'
            wandb.init(project='PeptideBERT', name=run_name)
            
            save_dir = f'./checkpoints/{run_name}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            shutil.copy('./config.yaml', f'{save_dir}/config.yaml')
            shutil.copy('./model/network.py', f'{save_dir}/network.py')
        else:
            save_dir = None
        
        # Train the model
        best_acc = train_model(model, train_data_loader, val_data_loader, optimizer, criterion, scheduler, config, save_dir)
        
        # Load best model and test
        if not config['debug']:
            model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)
        
        # Test the model
        test_acc, test_prec, test_rec, test_f1, y_true, y_pred = test(model, test_data_loader, device)
        print(f'\nTest Results for Run {run + 1}:')
        print(f'Test Accuracy: {test_acc}%')
        print(f'Test Precision: {test_prec}%')
        print(f'Test Recall: {test_rec}%')
        print(f'Test F1: {test_f1}%')
        
        # Save predictions for confusion matrix
        np.savez(f'test_predictions_run_{run + 1}.npz', 
                 y_true=np.array(y_true), 
                 y_pred=np.array(y_pred))
        
        # Save metrics for this run
        with open(f"test_metrics_run_{run + 1}.txt", "w") as f:
            f.write(f"# Test Metrics Log - Run {run + 1}\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Test Performance Metrics\n")
            f.write(f"- Accuracy: {test_acc}%\n")
            f.write(f"- Precision: {test_prec}%\n")
            f.write(f"- Recall: {test_rec}%\n")
            f.write(f"- F1 Score: {test_f1}%\n\n")
            f.write("## Model Parameters\n")
            f.write(f"- Learning Rate: {config['optim']['lr']}\n")
            f.write(f"- Batch Size: {config['batch_size']}\n")
            f.write(f"- Epochs: {config['epochs']}\n")
            f.write(f"- Scheduler: {config['sch']['name']}\n")
            if config['sch']['name'] == 'lronplateau':
                f.write(f"- Scheduler Factor: {config['sch']['factor']}\n")
                f.write(f"- Scheduler Patience: {config['sch']['patience']}\n")
        
        # Append to the all runs summary file
        with open("all_runs_summary.txt", "a") as all_runs_file:
            all_runs_file.write(f"## Run {run + 1}\n")
            all_runs_file.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            all_runs_file.write(f"- Model: {config['task']}-run{run+1}\n")
            all_runs_file.write(f"- Accuracy: {test_acc}%\n")
            all_runs_file.write(f"- Precision: {test_prec}%\n")
            all_runs_file.write(f"- Recall: {test_rec}%\n")
            all_runs_file.write(f"- F1 Score: {test_f1}%\n")
            all_runs_file.write(f"- Best Validation Accuracy: {best_acc}%\n\n")
        
        # Check if this is the best test performance
        if test_f1 > best_test_metrics['f1']:
            best_test_metrics = {
                'accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1': test_f1,
                'run': run + 1
            }
            
            # Save the best model parameters
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_metrics': best_test_metrics,
                'config': config
            }, 'checkpoints/best_test_model.pt')
            
            print(f'\nNew best test model saved! F1 Score: {test_f1}%')
            
    # Add best run information to the all runs summary
    with open("all_runs_summary.txt", "a") as all_runs_file:
        all_runs_file.write(f"# Best Run Summary\n")
        all_runs_file.write(f"- Best Run: {best_test_metrics['run']}\n")
        all_runs_file.write(f"- Best Accuracy: {best_test_metrics['accuracy']}%\n")
        all_runs_file.write(f"- Best Precision: {best_test_metrics['precision']}%\n")
        all_runs_file.write(f"- Best Recall: {best_test_metrics['recall']}%\n")
        all_runs_file.write(f"- Best F1 Score: {best_test_metrics['f1']}%\n")
        all_runs_file.write(f"- Model Location: checkpoints/best_test_model.pt\n")
    
    # Save test metrics in tabular format for all runs
    with open("all_test_metrics.txt", "w") as f:
        f.write(f"# All Test Metrics - Comparison Table\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write header
        f.write(f"| Run | Accuracy | Precision | Recall | F1 Score |\n")
        f.write(f"|-----|----------|-----------|--------|----------|\n")
        
        # Read all individual run files and extract metrics
        for run in range(1, 11):
            try:
                with open(f"test_metrics_run_{run}.txt", "r") as run_file:
                    lines = run_file.readlines()
                    acc_line = next((line for line in lines if "- Accuracy:" in line), "- Accuracy: 0%")
                    prec_line = next((line for line in lines if "- Precision:" in line), "- Precision: 0%")
                    rec_line = next((line for line in lines if "- Recall:" in line), "- Recall: 0%")
                    f1_line = next((line for line in lines if "- F1 Score:" in line), "- F1 Score: 0%")
                    
                    acc = acc_line.split(": ")[1].strip()
                    prec = prec_line.split(": ")[1].strip()
                    rec = rec_line.split(": ")[1].strip()
                    f1 = f1_line.split(": ")[1].strip()
                    
                    f.write(f"| {run} | {acc} | {prec} | {rec} | {f1} |\n")
            except FileNotFoundError:
                f.write(f"| {run} | N/A | N/A | N/A | N/A |\n")
        
        # Add empty line
        f.write("\n")
        
        # Mark the best run
        f.write(f"# Best Run: {best_test_metrics['run']}\n")
        f.write(f"- Accuracy: {best_test_metrics['accuracy']}%\n")
        f.write(f"- Precision: {best_test_metrics['precision']}%\n")
        f.write(f"- Recall: {best_test_metrics['recall']}%\n")
        f.write(f"- F1 Score: {best_test_metrics['f1']}%\n")
    
    print("All test metrics have been saved to all_test_metrics.txt in tabular format")
    
    # Save final best test metrics summary
    with open("best_test_metrics_summary.txt", "w") as f:
        f.write(f"# Best Test Model Summary\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Best Test Performance Metrics\n")
        f.write(f"- Accuracy: {best_test_metrics['accuracy']}%\n")
        f.write(f"- Precision: {best_test_metrics['precision']}%\n")
        f.write(f"- Recall: {best_test_metrics['recall']}%\n")
        f.write(f"- F1 Score: {best_test_metrics['f1']}%\n\n")
        f.write(f"## Best Model Information\n")
        f.write(f"- Run Number: {best_test_metrics['run']}\n")
        f.write(f"- Model Location: checkpoints/best_test_model.pt\n")
        f.write("\n## Model Configuration\n")
        f.write(f"- Learning Rate: {config['optim']['lr']}\n")
        f.write(f"- Batch Size: {config['batch_size']}\n")
        f.write(f"- Epochs: {config['epochs']}\n")
        f.write(f"- Scheduler: {config['sch']['name']}\n")
        if config['sch']['name'] == 'lronplateau':
            f.write(f"- Scheduler Factor: {config['sch']['factor']}\n")
            f.write(f"- Scheduler Patience: {config['sch']['patience']}\n")
    
    print(f'\nTraining completed! Best test model saved at: checkpoints/best_test_model.pt')
    print(f'Best test metrics from run {best_test_metrics["run"]}:')
    print(f'Accuracy: {best_test_metrics["accuracy"]}%')
    print(f'Precision: {best_test_metrics["precision"]}%')
    print(f'Recall: {best_test_metrics["recall"]}%')
    print(f'F1 Score: {best_test_metrics["f1"]}%')

if __name__ == '__main__':
    main()
