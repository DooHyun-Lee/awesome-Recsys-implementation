#  Implementation of the paper 'Self-Attentive Sequential Recommendation' (SASRec)

## Brief summary 
1. The input sequence consists solely of item IDs, followed by an item embedding table.<br/><br/>
2. Attention blocks are employed in an auto-regressive manner to predict the succeeding item at each timestep.<br/><br/>
3. The training loss function is formulated following the principles of contrastive learning, distinguishing between positive and negative items.<br/><br/>
4. During the inference stage, the results of sequence embedding are multiplied with item embeddings to capture the association between the item and the sequence. The ultimate rank is determined based on this outcome.<br/><br/>
5. In terms of evaluation, the study employed HT@10 and NDCG@10 metrics. An array including the real item and 100 randomly selected negative items is formed for evaluation. The final estimated rank is utilized for assessment.<br/><br/>

<br/><br/>

## :fire: Implementation details (Not mentioned in the paper)
* feed forward network consisted of Conv1d layer
  * result comparison with 2-layer mlp attatched below  <br/><br/>
* normalized query input for attention module
  * K, V inputs are not modified 

<br/><br/>

## :rocket: Experiment result
We used movieLens-1m dataset, pre-processing steps are included in the code <br/>
Below graph shows comparison between feed-forward-network with 2 layer mlp and 2 layer conv1d <br/>
Result with better loss value represents FFN with conv1d <br/>

<br/><br/>

## References 
1. Most of code implementation follows pytorch implementation of SASRec which could be found at  https://github.com/pmixer/SASRec.pytorch/tree/master <br/><br/>
2. Original paper could be found at https://arxiv.org/pdf/1808.09781.pdf