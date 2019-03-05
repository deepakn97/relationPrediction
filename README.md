# Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs

This program provides the implementation of the attention based model for the knowledge base completion task.

## Usage

### Requirements
- [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

Please download miniconda from above link and create an environment using the following command:

        `conda env create -f pytorch35.yml`

Activate the environment before executing the program as follows:

        `source activate pytorch35`

### Training
To run the program:
        
        python3 train.py --embedding_dim <int> --num_filters <int> --learning_rate <float> --name <dataset_name> [--useConstantInit] --model_name <name_of_saved_model>

**Required parameters:** 

`--data`: Specify the folder of the dataset. Default: Wordnet 

`--epochs`: Number of filters. Default: 3600

`--learning_rate`: Initial learning rate.

`--name`: Dataset name (WN18RR or FB15k-237).

`--useConstantInit`: Initialize filters by [0.1, 0.1, -0.1]. Otherwise, initialize filters by a truncated normal distribution.

`--model_name`: Name of saved models.

**Optional parameters:** 

`--l2_reg_lambda`: L2 regularizaion lambda (Default: 0.001).
  
`--dropout_keep_prob`: Dropout keep probability (Default: 1.0).
  
`--num_epochs`: Number of training epochs (Default: 200).

`--run_folder`: Specify directory path to save trained models.

`--batch_size`: Batch size.

### Reproduce the ConvKB results 

To reproduce the ConvKB results published in the paper:      
                
        $ python train.py --embedding_dim 100 --num_filters 50 --learning_rate 0.000005 --name FB15k-237 --useConstantInit --model_name fb15k237
        
        $ python train.py --embedding_dim 50 --num_filters 500 --learning_rate 0.0001 --name WN18RR --model_name wn18rr --saveStep 50
        
### Evaluation metrics

File `eval.py` provides ranking-based scores as evaluation metrics, including the mean rank, the mean reciprocal rank and Hits@10 in a setting protocol "Filtered".

Files `evalFB15k-237.sh` and `evalWN18RR.sh` contain evaluation commands. Depending on the memory resources, you should change the value of `--num_splits` to a suitable value to get a faster process. To get the results (supposing `num_splits = 8`):
        
        $ python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_name fb15k237 --num_splits 8 --decode
        
        $ python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --model_name wn18rr --num_splits 8 --decode
         
## Acknowledgments     

I would like to thank Denny Britz for implementing a CNN for text classification in TensorFlow.
