# POS Tagger using PyTorch and UD_English-Atis Dataset

This POS tagger is designed to classify parts-of-speech in English sentences using the UD_English-Atis dataset. The model is implemented in PyTorch and is based on a LSTM architecture. 

### Usage 
The POS tagger can be used for either training or prediction. To use it for prediction, simply run the script without any arguments:
`python pos_tagger.py`

This will load a pre-trained model called Final.pt from the same location as the script and prompt you for input sentences. You can enter one sentence, and the model will output the predicted part-of-speech tags for each token.

If the Final.pt file does not exist in the same location as the script, the model will automatically start training using the UD_English-Atis dataset. 
# Results
The model achieved an overall accuracy of 98% on the test set of the UD_English-Atis dataset. The precision, recall, and F1-score for each part-of-speech category are shown in the table below.
```
              precision    recall  f1-score   support

       <bot>       1.00      1.00      1.00       586
       <eot>       1.00      1.00      1.00       586
         adj       0.93      0.96      0.94       220
         adp       0.99      1.00      1.00      1434
         adv       0.95      0.78      0.86        76
         aux       0.97      0.99      0.98       256
       cconj       1.00      1.00      1.00       109
         det       0.99      0.87      0.93       512
        intj       1.00      1.00      1.00        36
        noun       0.99      0.99      0.99      1166
         num       0.97      0.88      0.92       127
        part       0.96      0.98      0.97        56
        pron       0.86      0.98      0.92       392
       propn       0.98      1.00      0.99      1567
        verb       0.99      0.98      0.98       629

    accuracy                           0.98      7752
   macro avg       0.97      0.96      0.96      7752
weighted avg       0.98      0.98      0.98      7752

```