#  Implementation of the paper 'Self-Attentive Sequential Recommendation' (SASRec)

## Brief summary 
1. The input sequence consists solely of item IDs, followed by an item embedding table.
2. Attention blocks are employed in an auto-regressive manner to predict the succeeding item at each timestep.
3. The training loss function is formulated following the principles of contrastive learning, distinguishing between positive and negative items.
4. During the inference stage, the results of sequence embedding are multiplied with item embeddings to capture the association between the item and the sequence. The ultimate rank is determined based on this outcome.
5. In terms of evaluation, the study employed HT@10 and NDCG@10 metrics. An array including the real item and 100 randomly selected negative items is formed for evaluation. The final estimated rank is utilized for assessment.

<br/><br/>

## :fire: Implementation details (Not mentioned in the paper)
* feed forward network consisted of Conv1d layer
  * result comparison attatched below  <br/><br/>
* normalized query input for attention module
  * K, V inputs are not modified 

<br/><br/>

## :rocket: Experiment result
