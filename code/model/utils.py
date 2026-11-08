import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(model, dataloader, optimizer, criterion, scheduler, device):
    """
    Training loop for refactored PeptideBERT.
    
    Note: criterion parameter is kept for API compatibility but is not used.
    Loss is computed inside model.forward() when labels are provided.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Model returns dict with 'loss', 'logits', 'probs'
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs['loss']

        loss.backward()
        optimizer.step()
        
        # Step the scheduler if it's OneCycleLR (which needs to be stepped every batch)
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validation loop for refactored PeptideBERT.
    
    Note: criterion parameter is kept for API compatibility but is not used.
    Loss is computed inside model.forward() when labels are provided.
    """
    model.eval()
    total_loss = 0.0

    ground_truth = []
    predictions = []

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.inference_mode():
            # Model returns dict with 'loss', 'logits', 'probs'
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            probs = outputs['probs'].squeeze(1)  # [batch_size]

        total_loss += loss.item()
    
        # Use 0.5 threshold on probabilities
        preds = torch.where(probs > 0.5, 1, 0)
        predictions.extend(preds.cpu().tolist())
        ground_truth.extend(labels.cpu().tolist())

    total_loss = total_loss / len(dataloader)
    accuracy = 100 * accuracy_score(ground_truth, predictions)
    precision = 100 * precision_score(ground_truth, predictions)
    recall = 100 * recall_score(ground_truth, predictions)
    f1 = 100 * f1_score(ground_truth, predictions)

    return total_loss, accuracy, precision, recall, f1


def test(model, dataloader, device):
    """
    Test loop for refactored PeptideBERT.
    Returns: accuracy, precision, recall, f1, ground_truth, predictions
    """
    model.eval()

    ground_truth = []
    predictions = []

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        with torch.inference_mode():
            # Model returns dict with 'logits', 'probs'
            outputs = model(input_ids, attention_mask)
            probs = outputs['probs'].squeeze(1)  # [batch_size]
    
        # Use 0.5 threshold on probabilities
        preds = torch.where(probs > 0.5, 1, 0)
        predictions.extend(preds.cpu().tolist())
        ground_truth.extend(labels.tolist())

    accuracy = 100 * accuracy_score(ground_truth, predictions)
    precision = 100 * precision_score(ground_truth, predictions)
    recall = 100 * recall_score(ground_truth, predictions)
    f1 = 100 * f1_score(ground_truth, predictions)

    return accuracy, precision, recall, f1, ground_truth, predictions
