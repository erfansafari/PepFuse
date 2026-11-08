import torch
from transformers import BertModel, BertConfig, logging
from typing import Dict, Optional

logging.set_verbosity_error()


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over sequence tokens, masking out padding tokens.
    
    Args:
        last_hidden_state: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len] with 1 for valid tokens, 0 for padding
    
    Returns:
        pooled: [batch_size, hidden_size]
    """
    # Expand mask to match hidden_state dimensions: [batch_size, seq_len, hidden_size]
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum of valid (non-masked) embeddings
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    
    # Sum of mask (number of valid tokens per sample)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
    
    # Mean pooling
    return sum_embeddings / sum_mask


class PeptideBERT(torch.nn.Module):
    """
    PeptideBERT for binary classification (AMP prediction).
    
    Key features:
    - Uses mean pooling over last_hidden_state instead of pooler_output
    - Outputs raw logits (no sigmoid in forward)
    - Returns dict with loss, logits, and probs
    - Supports two-stage training: head-only then fine-tune last layers
    """
    
    def __init__(self, bert_config):
        super(PeptideBERT, self).__init__()

        # Load pretrained ProtBERT
        # Note: ignore_mismatched_sizes=True allows loading even if hidden_size differs
        # from original checkpoint (1024). Mismatched layers will be randomly initialized
        # and require downstream training to learn task-specific representations.
        self.protbert = BertModel.from_pretrained(
            'Rostlab/prot_bert_bfd',
            config=bert_config,
            ignore_mismatched_sizes=True
        )
        
        # Classification head: outputs raw logits (no sigmoid)
        self.head = torch.nn.Linear(bert_config.hidden_size, 1)
        
        # Loss function (numerically stable)
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with improved API.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] optional, float values 0 or 1
        
        Returns:
            dict with keys:
                - 'logits': [batch_size, 1] raw logits
                - 'probs': [batch_size, 1] sigmoid probabilities
                - 'loss': scalar (only if labels provided)
        """
        # Get BERT outputs
        output = self.protbert(input_ids, attention_mask=attention_mask, return_dict=True)
        
        # Mean pooling over valid tokens
        pooled = mean_pooling(output.last_hidden_state, attention_mask)
        
        # Classification head (raw logits)
        logits = self.head(pooled)  # [batch_size, 1]
        
        # Compute probabilities for metrics/reporting
        probs = torch.sigmoid(logits)
        
        # Build output dict
        result = {
            'logits': logits,
            'probs': probs
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Ensure labels are float and have correct shape
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1).float()
            else:
                labels = labels.float()
            
            loss = self.criterion(logits, labels)
            result['loss'] = loss
        
        return result


def set_training_stage(model: PeptideBERT, stage: str, n_layers: int = 4) -> None:
    """
    Configure model for two-stage training.
    
    Stage A (head-only):
        - Freeze all ProtBERT parameters
        - Train only the classification head
        - Use higher learning rate (~1e-3)
    
    Stage B (fine-tune):
        - Unfreeze last N encoder layers + head
        - Use differential learning rates:
            * head: ~1e-4
            * encoder layers: ~2e-5
    
    Args:
        model: PeptideBERT model
        stage: 'A' for head-only, 'B' for fine-tuning
        n_layers: number of last encoder layers to unfreeze in stage B
    """
    if stage.upper() == 'A':
        # Stage A: Freeze all ProtBERT, train only head
        print(f"[Stage A] Freezing all ProtBERT parameters. Training head only.")
        
        for param in model.protbert.parameters():
            param.requires_grad = False
        
        for param in model.head.parameters():
            param.requires_grad = True
            
    elif stage.upper() == 'B':
        # Stage B: Unfreeze last N encoder layers + head
        print(f"[Stage B] Unfreezing last {n_layers} encoder layers + head for fine-tuning.")
        
        # First, freeze everything
        for param in model.protbert.parameters():
            param.requires_grad = False
        
        # Unfreeze last N encoder layers
        total_layers = len(model.protbert.encoder.layer)
        unfrozen_layers = model.protbert.encoder.layer[-n_layers:]
        
        for layer in unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Unfreeze head
        for param in model.head.parameters():
            param.requires_grad = True
        
        print(f"  - Unfrozen encoder layers: {total_layers - n_layers} to {total_layers - 1}")
        print(f"  - Total encoder layers: {total_layers}")
    
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'A' or 'B'.")
    
    # Print trainable parameters summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")


def create_model(config):
    """Create PeptideBERT model with configuration."""
    bert_config = BertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['network']['hidden_size'],
        num_hidden_layers=config['network']['hidden_layers'],
        num_attention_heads=config['network']['attn_heads'],
        hidden_dropout_prob=config['network']['dropout']
    )
    model = PeptideBERT(bert_config).to(config['device'])
    
    return model


def cri_opt_sch(config, model, stage: str = 'A'):
    """
    Create criterion, optimizer, and scheduler for training.
    
    Args:
        config: training configuration dict
        model: PeptideBERT model
        stage: 'A' for head-only training, 'B' for fine-tuning with differential LR
    
    Returns:
        criterion, optimizer, scheduler
    """
    # Criterion is now inside the model, but we return None to maintain API compatibility
    # The loss is computed in model.forward() when labels are provided
    criterion = None  # Loss computed inside model
    
    # Configure optimizer based on training stage
    if stage.upper() == 'A':
        # Stage A: Head-only training with single learning rate
        lr = config['optim'].get('lr', 1e-3)
        
        # Only include trainable parameters (head only)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=config['optim'].get('weight_decay', 1e-5)
        )
        
        print(f"[Optimizer] Stage A - Single LR: {lr}")
    
    elif stage.upper() == 'B':
        # Stage B: Fine-tuning with differential learning rates
        head_lr = config['optim'].get('head_lr', 1e-4)
        encoder_lr = config['optim'].get('encoder_lr', 2e-5)
        
        # Separate parameter groups
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        encoder_params = [p for p in model.protbert.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': head_lr}
        ], weight_decay=config['optim'].get('weight_decay', 1e-5))
        
        print(f"[Optimizer] Stage B - Encoder LR: {encoder_lr}, Head LR: {head_lr}")
    
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'A' or 'B'.")
    
    # Scheduler configuration (works with both single and multiple param groups)
    scheduler = None
    
    if config['sch']['name'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config['optim'].get('encoder_lr', config['optim']['lr']), 
                    config['optim'].get('head_lr', config['optim']['lr'])] if stage.upper() == 'B' 
                   else config['optim']['lr'],
            epochs=config['epochs'],
            steps_per_epoch=config['sch']['steps']
        )
    elif config['sch']['name'] == 'lronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['sch']['factor'],
            patience=config['sch']['patience']
        )
    else:
        print(f"Warning: Unknown scheduler '{config['sch']['name']}'. Using default ReduceLROnPlateau.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5
        )
    
    return criterion, optimizer, scheduler
