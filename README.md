# EEGNN

This is a repo that reproduce the result of E-ReaRev

## Run
Before you run the code, you must download the data from https://drive.google.com/file/d/1p7eLSsSKkZQxB32mT5lMsthVP6R_3x1j/view
and unzip to this folder

And make sure you have the requirement need in requirement.txt


### Train
You can train you own model by following scripts
python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 3 --relationi_word_emb True --name webqsp

### Eval
After training, there will be a [name]final.ckpt generate in the checkpoint/pretrained folder. Rename it as a name you want. E.g train.ckpt

To evaluate the model

python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 3 --relationi_word_emb True --name webqsp --load__experiment train.ckpt --is_eval

#### In order to choose the % of KG incompletess
Modify the fact_dropout in the line 159 of evaluate.py 
e.g 0.1 means KG has 10% edges be erased

#### In order to use the EE extend module
Uncommented the line 267-273 in models/ReaRev/rearev.py

Run the same command above 

Modify the line 272 threshold to use different threshold

Uncommented the line 274 to update the local_entity_embedding

## Experiment Result
### Result of erase edge without EE
The result is in folder originalResult

### Result of erase edge with EE threshold 0.3
The result is in folder EEThreshold0.3

### Result of erase edge with EE threshold 0.7
The result is in folder EEThreshold0.7

### Result of erase edge with EE threshold 0.3 and update local embedding
The result is in folder LocalEmbeddingThreshold0.3
