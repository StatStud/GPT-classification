import io
import csv
import os
from ml_things import plot_dict, plot_confusion_matrix, fix_text
import torch
import torch.distributed as dist
import argparse
import pandas as pd
#from tqdm.notebook import tqdm
from tqdm import tqdm
import time
from datetime import datetime
import sys
from sklearn.metrics import confusion_matrix
from io import StringIO
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (set_seed,
             TrainingArguments,
             Trainer,
             GPT2Config,
             GPT2Tokenizer,
             AdamW, 
             get_linear_schedule_with_warmup,
             GPT2ForSequenceClassification)


def main(args):
  # Initialize distributed training environment
  dist.init_process_group(backend='nccl')
  gpu_count = dist.get_world_size()

  # Get the local rank for the current process
  local_rank = int(os.environ.get("LOCAL_RANK"))

  # Set the GPU device for the current process
  device = torch.device("cuda", local_rank)

  # Set the current GPU device as the default device
  torch.cuda.set_device(device)

  #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
  print("USING MODEL:",args.model) 
  model_name_or_path = args.model
  epochs = args.epochs
  learning_rate = args.lr
  batch_size = args.batch_size

  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################


  task = 'binary'
  #task = "binary"
  suffix = ""

  train_file = f"train.csv"
  val_file = f"val.csv"
  test_file = f"test.csv"

  # Set seed for reproducibility.
  set_seed(123)



  # Pad or truncate text sequences to a specific length
  # if `None` it will use maximum sequence of word piece tokens allowed by model.
  max_length = 60

  # Look for gpu to use. Will use `cpu` by default if no gpu found.
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Initialize distributed training environment
  ##dist.init_process_group(backend='nccl')
  ##device = torch.device('cuda', torch.cuda.current_device())

  # Name of transformers model - will use already pretrained model.
  # Path of transformer model - will load your own model from local disk.


  # Dictionary of labels and their id - this will be used to convert.
  # String labels to number ids.
  #labels_ids = {'neg': 0, 'pos': 1}
  if task == "binary":
    labels_ids = {'0': 0, '1': 1}
    n_labels = len(labels_ids)
  else:
    labels_ids = {str(i): i for i in range(60)}
    n_labels = 60

  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################

  class MasterDataset(Dataset):
    def __init__(self, path, use_tokenizer, task):
      self.texts = []
      self.labels = []

      df = pd.read_csv(path)
      df = df.dropna()
      ####################################################################################
      ################### Continue with the rest of the script as normal #################
      ####################################################################################

      # Assuming the text is in the first column
      content_column = df.loc[:, 'text']  
      if task == "binary":
        label_column = df.loc[:, 'label']
      else:
        label_column = df.loc[:, 'modified_label']

      # Iterate over the columns and append data to lists
      for content, label in zip(content_column, label_column):
        content = fix_text(content)
        self.texts.append(content)
        self.labels.append(int(label))

      # Number of examples.
      self.n_examples = len(self.labels)



    def __len__(self):
      return self.n_examples

    def __getitem__(self, item):
      return {'text':self.texts[item],
          'label':self.labels[item]}



  class Gpt2ClassificationCollator(object):

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

      # Tokenizer to be used inside the class.
      self.use_tokenizer = use_tokenizer
      # Check max sequence length.
      self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
      # Label encoder used inside the class.
      self.labels_encoder = labels_encoder

      # Check if labels_encoder is provided; if not, use default label encoder
      if labels_encoder is None:
        self.labels_encoder = {'0': 0, '1': 1}
      else:
        self.labels_encoder = labels_encoder

      return

    def __call__(self, sequences):

      # Get all texts from sequences list.
      texts = [sequence['text'] for sequence in sequences]
      # Get all labels from sequences list.
      labels = [sequence['label'] for sequence in sequences]
      # Encode all labels using label encoder.
      ##labels = [self.labels_encoder[label] for label in labels]
      # Call tokenizer on all texts to convert into tensors of numbers with 
      # appropriate padding.
      inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
      # Update the inputs with the associated encoded labels as tensor.
      inputs.update({'labels':torch.tensor(labels)})

      return inputs


  def train(dataloader, optimizer_, scheduler_, device_, model_):

    # Use global variable for model.
    #global model
    model = model_

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()


   # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):

      # Add original labels - use later for evaluation.
      true_labels += batch['labels'].numpy().flatten().tolist()

      # move batch to device
      batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

      # Always clear any previously calculated gradients before performing a
      # backward pass.
      model.zero_grad()

      # Perform a forward pass (evaluate the model on this training batch).
      # This will return the loss (rather than the model output) because we
      # have provided the `labels`.
      # The documentation for this a bert model function is here: 
      # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
      outputs = model(**batch)

      # The call to `model` always returns a tuple, so we need to pull the 
      # loss value out of the tuple along with the logits. We will use logits
      # later to calculate training accuracy.
      loss, logits = outputs[:2]

      # Accumulate the training loss over all of the batches so that we can
      # calculate the average loss at the end. `loss` is a Tensor containing a
      # single value; the `.item()` function just returns the Python value 
      # from the tensor.
      total_loss += loss.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # Update parameters and take a step using the computed gradient.
      # The optimizer dictates the "update rule"--how the parameters are
      # modified based on their gradients, the learning rate, etc.
      optimizer.step()

      # Update the learning rate.
      scheduler.step()

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()

      # Convert these logits to list of predicted labels values.
      predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss



  def validation(dataloader, device_, model_):

    # Use global variable for model.
    #global model
    model = model_

    # Tracking variables
    predictions_labels = []
    true_labels = []
    #total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()


    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):

      # Add original labels - use later for evaluation.
      true_labels += batch['labels'].numpy().flatten().tolist()

      # move batch to device
      batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():     

        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
        loss, logits = outputs[:2]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()

        # update list
        predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


  def test(dataloader, device_, model_):

    # Use global variable for model.
    #global model 
    all_evaluation_results = []
    model = model_
     
    # Tracking variables
    predictions_labels = []
    true_labels = []
    #total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()


    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):

      # Add original labels - use later for evaluation.
      true_labels += batch['labels'].numpy().flatten().tolist()

      # move batch to device
      batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():     

        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
        loss, logits = outputs[:2]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()

        # update list
        predictions_labels += predict_content

    # Gather predictions from all GPUs to rank 0
    all_predictions = torch.tensor(predictions_labels).to(device_)
    gathered_predictions = [torch.zeros_like(all_predictions) for _ in range(torch.distributed.get_world_size())]
    if local_rank == 0:
      dist.gather(all_predictions, gathered_predictions, dst=0)
    else:
      dist.gather(all_predictions, dst=0)

    # Use communication method to gather evaluation results from all processes (GPUs) on rank 0
    all_labels = torch.tensor(true_labels).to(device_)
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(torch.distributed.get_world_size())]
    if local_rank == 0:
      dist.gather(all_labels, gathered_labels, dst=0)
    else:
      dist.gather(all_labels, dst=0)

    # Aggregate evaluation results on the main process (rank 0)
    if local_rank == 0:
      if local_rank == 0:
      # Concatenate evaluation results from all GPUs
        gathered_predictions = torch.cat(gathered_predictions)
        gathered_labels = torch.cat(gathered_labels)

        # Move the results back to CPU
        all_labels = gathered_labels.cpu().tolist()
        all_predictions = gathered_predictions.cpu().tolist()
         
        # Calculate the confusion matrix
        confusion = confusion_matrix(all_labels, all_predictions)

        # Extract TP, FP, TN, and FN from the confusion matrix
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP = confusion[1, 1]
         
        print(TN,FP,FN,TP)

  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################

  # Get model configuration.
  print('Loading configuraiton...')
  model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

  # Get model's tokenizer.
  print('Loading tokenizer...')
  tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
  # default to left padding
  tokenizer.padding_side = "left"
  # Define PAD Token = EOS Token = 50256
  tokenizer.pad_token = tokenizer.eos_token


  # Get the actual model.
  print('Loading model...')
  model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
  model_size = model.num_parameters()
   
  # resize model embedding to match new tokenizer
  model.resize_token_embeddings(len(tokenizer))

  # fix model padding token id
  model.config.pad_token_id = model.config.eos_token_id
   
  # Load model to defined device.
  model.to(device)
  print('Model loaded to `%s`'%device)
   
  # Wrap the model with DistributedDataParallel
  model = torch.nn.parallel.DistributedDataParallel(model)



  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################

  # Create data collator to encode text and labels into numbers.
  gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                               labels_encoder=labels_ids, 
                               max_sequence_len=max_length)


  print('Dealing with Train...')
  # Create pytorch dataset.
  train_dataset = MasterDataset(path = train_file, use_tokenizer=tokenizer, task=task)
  print('Created `train_dataset` with %d examples!'%len(train_dataset))

  # Move pytorch dataset into dataloader.
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, collate_fn=gpt2_classificaiton_collator)
  print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

  print("\n")

  print('Dealing with Validation...')
  # Create pytorch dataset.
  valid_dataset = MasterDataset(path = val_file, use_tokenizer=tokenizer, task=task)
  print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

  # Move pytorch dataset into dataloader.
  valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
  valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, collate_fn=gpt2_classificaiton_collator)
  print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

  print("\n")

  print('Dealing with Test...')
  # Create pytorch dataset.
  test_dataset = MasterDataset(path = test_file, use_tokenizer=tokenizer, task=task)
  print('Created `valid_dataset` with %d examples!'%len(test_dataset))

  # Move pytorch dataset into dataloader.
  test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, collate_fn=gpt2_classificaiton_collator)
  print('Created `eval_dataloader` with %d batches!'%len(test_dataloader))

  # Synchronize the model and dataloaders across all processes
  torch.distributed.barrier()
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################

  # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
  # I believe the 'W' stands for 'Weight Decay fix"
  optimizer = AdamW(model.parameters(),
           lr = learning_rate, # default is 5e-5, our notebook had 2e-5
           eps = 1e-8 # default is 1e-8.
           )

  # Total number of training steps is number of batches * number of epochs.
  # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
  # us the number of batches.
  total_steps = len(train_dataloader) * epochs

  # Create the learning rate scheduler.
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                        num_warmup_steps = 0, # Default value in run_glue.py
                        num_training_steps = total_steps)

  # Store the average loss after each epoch so we can plot them.
  all_loss = {'train_loss':[], 'val_loss':[]}
  all_acc = {'train_acc':[], 'val_acc':[]}
   
  # Initialize an empty list to store evaluation results from each GPU
  all_evaluation_results = []

  # Loop through each epoch.
  tic = time.time()
  print('Epoch')
  for epoch in tqdm(range(epochs)):
    print()
    print(f"starting on {epoch} out of {epochs}")
    print('Training on batches...')
     
    # Set the epoch for the dataloaders
    train_dataloader.sampler.set_epoch(epoch)
    valid_dataloader.sampler.set_epoch(epoch)
    test_dataloader.sampler.set_epoch(epoch)
     
    # Perform one full pass over the training set.
    train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device, model)
    train_acc = accuracy_score(train_labels, train_predict)

    # Get prediction form model on validation data. 
    print('Validation on batches...')
    valid_labels, valid_predict, val_loss = validation(valid_dataloader, device, model)
    val_acc = accuracy_score(valid_labels, valid_predict)

    # Print loss and accuracy values to see how training evolves.
    print(" train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
    print()

    # Store the loss value for plotting the learning curve.
    all_loss['train_loss'].append(train_loss)
    all_loss['val_loss'].append(val_loss)
    all_acc['train_acc'].append(train_acc)
    all_acc['val_acc'].append(val_acc)
    
  toc = time.time()
  elapsed_time = toc - tic
  elapsed_time = elapsed_time / 60
  formatted_elapsed_time = "{:.2f}".format(elapsed_time)

    
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################
  ##################################################################################################################

  #accuracy,weighted_precision,weighted_recall,weighted_f1_score = test(test_dataloader, device, model)
  # Instead of unpacking the return value into variables, store it in a variable
   
  # Redirect standard output to a StringIO object
  output_catcher = StringIO()
  sys.stdout = output_catcher
  test(test_dataloader, device, model)
   
  # Get the printed output from the StringIO object
  printed_output = output_catcher.getvalue()

  # Reset standard output to the original value
  sys.stdout = sys.__stdout__
   
  # Split the input string by space to get individual values
  values = printed_output.split()

  def extract_values_from_string(values):

    # Convert the values to integers and assign them to TN, FP, FN, TP
    TN, FP, FN, TP = map(int, values)

    return TN, FP, FN, TP
   
  TN = "fail"
  FP ="fail"
  FN = "fail"
  TP = "fail"
  precision = "fail"
  recall = "fail"
  f1 = "fail"
  accuracy = "fail"
   
  if values:
    TN, FP, FN, TP = extract_values_from_string(values)
   
    # Compute Precision
    precision = round(TP / (TP + FP),4)

    # Compute Recall
    recall = round(TP / (TP + FN),4)

    # Compute F1-Score
    f1 = round(2 * (precision * recall) / (precision + recall),4)

    # Compute accuracy
    accuracy = round((TP + TN) / (TP + TN + FP + FN),4)

    # Print the extracted values
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
   

  # Get the current datetime object
  current_datetime = datetime.now()

  # Convert the datetime object to a string in a specific format
  timestamp_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
   
  # Prepare the data as a dictionary
  data = {
    "Model": [args.model],
    "Task": ["Binary"],
    "Training_time": [formatted_elapsed_time],
    "GPU_count": [gpu_count],
    "model_size": [model_size],
    "learning_rate": [args.lr],
    "batch_size": [args.batch_size],
    "epochs": [args.epochs],
    "accuracy": [accuracy],
    "TP": [TP],
    "TN": [TN],
    "FP": [FP],
    "FN": [FN],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "date_time": [timestamp_string]
  }

  # Convert the data dictionary to a pandas DataFrame
  df = pd.DataFrame(data)
  df.dropna(inplace=True)
  df = df[df['TP'] != 'fail']
   
  obs_count = df["TN"] + df["TP"] + df["FN"] + df["FP"]
  df.insert(11, "obs_count", obs_count)
   
  csv_file = "model_tracking.csv"
   
  # If the file does not exist, create it and write the DataFrame
  if not os.path.isfile(csv_file):
    df.to_csv(csv_file, index=False)
  else:
    # If the file already exists, append the DataFrame without writing the header again
    with open(csv_file, 'a', newline='') as file:
      df.to_csv(file, index=False, header=False)
   

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script description')
  parser.add_argument('--lr', type=float, default=0.00002, help='Learning rate')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
  parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
  parser.add_argument('--model', type=str, default='gpt2', help='Model name or path')
  parser.add_argument('--nproc_per_node', type=int, help='Num of GPUs')
  # Add any other command-line arguments you need
  # ...

  args, _ = parser.parse_known_args()

  main(args)
