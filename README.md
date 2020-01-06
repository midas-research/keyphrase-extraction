# keyphrase-extraction


# Sequence Labelling with BiLSTM-CNN-CRF

 Instructions 
 <ol>
  <li> Use pytorch==1.1.0 and Include the above flair folder in your code repo. Do not use flair downloaded using pip </li>
  <li> Run test.py to check for NER task using CoNLL 2003 Dataset </li>
  </ol>
  To import character embeddings 
  
```python
from flair.embeddings import CharacterCNNEmbeddings
embeddings = CharacterCNNEmbeddings()
```

## Architectures 
The following architectures have been tried out for the experiments.
```python
from flair.models.sequence_tagger_CNN import SequenceTagger_CNN
# word embeddings -> cnn -> maxpool -> CRF 
tagger: SequenceTagger_CNN = SequenceTagger_CNN(hidden_size=200,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
# word embeddings -> muti-channel cnns -> maxpool -> CRF
tagger: SequenceTagger_CNN = SequenceTagger_CNN(hidden_size=200,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True,
                                        use_multichannels=True)
                                        
# word embeddings -> cnn -> maxpool -> LSTM -> CRF 
from flair.models.sequence_tagger_combo import SequenceTagger
#from flair.models import SequenceTagger
tagger: SequenceTagger = SequenceTagger(hidden_size=200,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
```
<ol>
 <li>  <b> word embeddings -> cnn -> maxpool -> CRF </b> </li>-
  <li>  <b> word embeddings -> muti-channel cnns -> maxpool -> CRF </b> </li>
  <li>  <b> word embeddings -> cnn -> maxpool -> blstm -> CRF </b> </li>
 </ol>

