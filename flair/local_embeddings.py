# import math
# import random
# from collections import Counter
#
# import torch
# from typing import List, Union
#
# import flair
# from flair.data import Sentence
# from flair.embeddings import TokenEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentEmbeddings
# from flair.hyperparameter import Parameter
# from flair.nn import LockedDropout, WordDropout
# import time
#
#
# class Stopwatch(object):
#     """A stopwatch utility for timing execution that can be used as a regular
#     object or as a context manager.
#     NOTE: This should not be used an accurate benchmark of Python code, but a
#     way to check how much time has elapsed between actions. And this does not
#     account for changes or blips in the system clock.
#     Instance attributes:
#     start_time -- timestamp when the timer started
#     stop_time -- timestamp when the timer stopped
#     As a regular object:
#     >>> stopwatch = Stopwatch()
#     >>> stopwatch.start()
#     >>> time.sleep(1)
#     >>> 1 <= stopwatch.time_elapsed <= 2
#     True
#     >>> time.sleep(1)
#     >>> stopwatch.stop()
#     >>> 2 <= stopwatch.total_run_time
#     True
#     As a context manager:
#     >>> with Stopwatch() as stopwatch:
#     ...     time.sleep(1)
#     ...     print repr(1 <= stopwatch.time_elapsed <= 2)
#     ...     time.sleep(1)
#     True
#     >>> 2 <= stopwatch.total_run_time
#     True
#     """
#
#     def __init__(self):
#         """Initialize a new `Stopwatch`, but do not start timing."""
#         self.start_time = None
#         self.stop_time = None
#
#     def start(self):
#         """Start timing."""
#         self.start_time = time.time()
#
#     def stop(self):
#         """Stop timing."""
#         self.stop_time = time.time()
#
#     @property
#     def time_elapsed(self):
#         """Return the number of seconds that have elapsed since this
#         `Stopwatch` started timing.
#         This is used for checking how much time has elapsed while the timer is
#         still running.
#         """
#         assert not self.stop_time, \
#             "Can't check `time_elapsed` on an ended `Stopwatch`."
#         return time.time() - self.start_time
#
#     @property
#     def total_run_time(self):
#         """Return the number of seconds that elapsed from when this `Stopwatch`
#         started to when it ended.
#         """
#         return self.stop_time - self.start_time
#
#     def __enter__(self):
#         """Start timing and return this `Stopwatch` instance."""
#         self.start()
#         return self
#
#     def __exit__(self, type, value, traceback):
#         """Stop timing.
#         If there was an exception inside the `with` block, re-raise it.
#         >>> with Stopwatch() as stopwatch:
#         ...     raise Exception
#         Traceback (most recent call last):
#             ...
#         Exception
#         """
#         self.stop()
#
#
# # class DocumentRNNEmbeddingsFast(DocumentEmbeddings):
# #     def __init__(
# #             self,
# #             embeddings: List[TokenEmbeddings],
# #             hidden_size=128,
# #             rnn_layers=1,
# #             reproject_words: bool = True,
# #             reproject_words_dimension: int = None,
# #             bidirectional: bool = False,
# #             dropout: float = 0.5,
# #             word_dropout: float = 0.0,
# #             locked_dropout: float = 0.0,
# #             rnn_type="GRU",
# #     ):
# #         """The constructor takes a list of embeddings to be combined.
# #         :param embeddings: a list of token embeddings
# #         :param hidden_size: the number of hidden states in the rnn
# #         :param rnn_layers: the number of layers for the rnn
# #         :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
# #         layer before putting them into the rnn or not
# #         :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
# #         dimension as before will be taken.
# #         :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
# #         :param dropout: the dropout value to be used
# #         :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
# #         :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
# #         :param rnn_type: 'GRU' or 'LSTM'
# #         """
# #         super().__init__()
# #
# #         self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
# #
# #         self.rnn_type = rnn_type
# #
# #         self.reproject_words = reproject_words
# #         self.bidirectional = bidirectional
# #
# #         self.length_of_all_token_embeddings: int = self.embeddings.embedding_length
# #
# #         self.static_embeddings = False
# #
# #         self.__embedding_length: int = hidden_size
# #         if self.bidirectional:
# #             self.__embedding_length *= 4
# #
# #         self.embeddings_dimension: int = self.length_of_all_token_embeddings
# #         if self.reproject_words and reproject_words_dimension is not None:
# #             self.embeddings_dimension = reproject_words_dimension
# #
# #         self.word_reprojection_map = torch.nn.Linear(
# #             self.length_of_all_token_embeddings, self.embeddings_dimension
# #         )
# #
# #         # bidirectional RNN on top of embedding layer
# #         if rnn_type == "LSTM":
# #             self.rnn = torch.nn.LSTM(
# #                 self.embeddings_dimension,
# #                 hidden_size,
# #                 num_layers=rnn_layers,
# #                 bidirectional=self.bidirectional,
# #             )
# #         else:
# #             self.rnn = torch.nn.GRU(
# #                 self.embeddings_dimension,
# #                 hidden_size,
# #                 num_layers=rnn_layers,
# #                 bidirectional=self.bidirectional,
# #             )
# #
# #         self.name = "document_" + self.rnn._get_name()
# #
# #         # dropouts
# #         if locked_dropout > 0.0:
# #             self.dropout: torch.nn.Module = LockedDropout(locked_dropout)
# #         else:
# #             self.dropout = torch.nn.Dropout(dropout)
# #
# #         self.use_word_dropout: bool = word_dropout > 0.0
# #         if self.use_word_dropout:
# #             self.word_dropout = WordDropout(word_dropout)
# #
# #         torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)
# #
# #         self.to(flair.device)
# #
# #     @property
# #     def embedding_length(self) -> int:
# #         return self.__embedding_length
# #
# #     def embed(self, sentences: Union[List[Sentence], Sentence]):
# #         """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
# #          only if embeddings are non-static."""
# #
# #         if type(sentences) is Sentence:
# #             sentences = [sentences]
# #
# #         self.rnn.zero_grad()
# #
# #         sentences.sort(key=lambda x: len(x), reverse=True)
# #
# #         self.embeddings.embed(sentences)
# #
# #         # first, sort sentences by number of tokens
# #         longest_token_sequence_in_batch: int = len(sentences[0])
# #
# #         all_sentence_tensors = []
# #         lengths: List[int] = []
# #
# #         # go through each sentence in batch
# #         for i, sentence in enumerate(sentences):
# #             lengths.append(len(sentence.tokens))
# #
# #             sentence_tensor = torch.cat([token.get_embedding().unsqueeze(0) for token in sentence], 0).to(flair.device)
# #
# #             # ADD TO SENTENCE LIST: add the representation
# #             all_sentence_tensors.append(sentence_tensor)
# #
# #         # --------------------------------------------------------------------
# #         # FF PART
# #         # --------------------------------------------------------------------
# #         # use word dropout if set
# #         # if self.use_word_dropout:
# #         #     sentence_tensor = self.word_dropout(sentence_tensor)
# #
# #         # if self.reproject_words:
# #         #     sentence_tensor = self.word_reprojection_map(sentence_tensor)
# #
# #         # sentence_tensor = self.dropout(sentence_tensor)
# #
# #         packed = torch.nn.utils.rnn.pack_sequence(all_sentence_tensors, lengths)
# #
# #         self.rnn.flatten_parameters()
# #
# #         rnn_out, hidden = self.rnn(packed)
# #
# #         outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_out)
# #
# #         outputs = self.dropout(outputs)
# #
# #         # --------------------------------------------------------------------
# #         # EXTRACT EMBEDDINGS FROM RNN
# #         # --------------------------------------------------------------------
# #         for sentence_no, length in enumerate(lengths):
# #             last_rep = outputs[length - 1, sentence_no]
# #
# #             embedding = last_rep
# #             if self.bidirectional:
# #                 first_rep = outputs[0, sentence_no]
# #                 embedding = torch.cat([first_rep, last_rep], 0)
# #
# #             sentence = sentences[sentence_no]
# #             sentence.set_embedding(self.name, embedding)
# #
# #     def _add_embeddings_internal(self, sentences: List[Sentence]):
# #         pass
#
#
# # class MemoryEmbeddingsLocal(TokenEmbeddings):
# #     def __init__(
# #             self,
# #             contextual_embeddings: TokenEmbeddings,
# #             concat_word_embeddings: bool = True,
# #             hidden_states: int = 64,
# #             reproject: bool = True,
# #             dropout: float = 0.5,
# #             word_dropout: float = 0.0,
# #             local_dropout: float = 0.1,
# #             use_batch_memory_state: bool = False,
# #             rnn_type: str = 'LSTM',
# #             train_initial_hidden_state: bool = True,
# #     ):
# #
# #         super().__init__()
# #
# #         # variables
# #         self.hidden_states = hidden_states
# #         self.concat_word_embeddings: bool = concat_word_embeddings
# #         self.word_dropout: float = word_dropout
# #         self.static_embeddings: bool = False
# #
# #         self.context_embeddings: TokenEmbeddings = contextual_embeddings
# #
# #         self.sub_embedding_names = (
# #             [emb.name for emb in self.context_embeddings.embeddings]
# #             if type(self.context_embeddings) is StackedEmbeddings
# #             else [self.context_embeddings.name]
# #         )
# #
# #         state_length = self.hidden_states if not use_batch_memory_state else self.hidden_states * 2
# #
# #         # the memory
# #         self.state_history = {}
# #
# #         # the NN
# #         self.dropout = torch.nn.Dropout(dropout)
# #         self.local_dropout = torch.nn.Dropout(local_dropout)
# #
# #         self.reproject: bool = reproject
# #         if self.reproject:
# #             self.reprojection_layer = torch.nn.Linear(
# #                 self.context_embeddings.embedding_length,
# #                 self.context_embeddings.embedding_length,
# #             )
# #
# #         self.rnn_type = rnn_type
# #
# #         if self.rnn_type == 'GRU':
# #             self.rnn = torch.nn.GRU(
# #                 self.context_embeddings.embedding_length,
# #                 self.hidden_states,
# #                 num_layers=1,
# #                 bidirectional=False,
# #             )
# #             # default zero-state tensor
# #             self.initial_hidden = torch.zeros(self.hidden_states, device=flair.device)
# #
# #             # make initial hidden state trainable if set
# #             if train_initial_hidden_state:
# #                 self.initial_hidden = torch.nn.Parameter(
# #                     self.initial_hidden,
# #                     requires_grad=True,
# #                 )
# #
# #         if self.rnn_type == 'LSTM':
# #             self.rnn = torch.nn.LSTM(
# #                 self.context_embeddings.embedding_length,
# #                 self.hidden_states,
# #                 num_layers=1,
# #                 bidirectional=False,
# #             )
# #             # default zero-state tensor
# #             self.lstm_init_h = torch.zeros(self.hidden_states, device=flair.device)
# #             self.lstm_init_c = torch.zeros(self.hidden_states, device=flair.device)
# #
# #             # make initial hidden state trainable if set
# #             if train_initial_hidden_state:
# #                 self.lstm_init_h = torch.nn.Parameter(torch.zeros(self.hidden_states, device=flair.device),
# #                                        requires_grad=True, )
# #
# #                 self.lstm_init_c = torch.nn.Parameter(torch.zeros(self.hidden_states, device=flair.device),
# #                                        requires_grad=True, )
# #
# #         self.name = self.context_embeddings.name + "-memory"
# #
# #         self.use_batch_memory_state = use_batch_memory_state
# #
# #         # determine embedding length
# #         dummy = Sentence('dummysent')
# #         self.embed(dummy)
# #         embedding = dummy[0].embedding
# #         # print(embedding)
# #         self.__embedding_length = len(embedding)
# #         self.eval()
# #
# #     def train(self, mode=True):
# #         super().train(mode=mode)
# #         if mode:
# #             # memory is wiped each time we do a training run
# #             print("train mode resetting embeddings")
# #             self.state_history = {}
# #             print(self.state_history)
# #
# #     def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
# #
# #         self.context_embeddings.embed(sentences)
# #
# #         # determine and add to history of each token
# #         counter_of_surface_forms_in_batch = Counter()
# #         surface_form_history = {}
# #         for sentence in sentences:
# #             for token in sentence:
# #                 counter_of_surface_forms_in_batch[token.text] += 1
# #                 # add current embedding to the memory
# #                 local_embedding = token.get_subembedding(
# #                     self.sub_embedding_names
# #                 )
# #
# #                 # drop the local embedding
# #                 self.local_dropout(local_embedding)
# #
# #                 if token.text not in surface_form_history:
# #                     surface_form_history[token.text] = [local_embedding]
# #                 else:
# #                     surface_form_history[token.text].append(local_embedding)
# #
# #         # sort surface forms by longest history
# #         surface_forms_sorted_by_memory_length = sorted(surface_form_history,
# #                                                        key=lambda k: len(surface_form_history[k]),
# #                                                        reverse=True)
# #         logest_memory_length_in_batch = len(surface_form_history[surface_forms_sorted_by_memory_length[0]])
# #
# #         # initialize zero-padded word embeddings tensor
# #         all_surface_form_tensor = torch.zeros(
# #             [
# #                 logest_memory_length_in_batch,
# #                 len(surface_form_history),
# #                 self.context_embeddings.embedding_length,
# #             ],
# #             dtype=torch.float,
# #             device=flair.device,
# #         )
# #
# #         all_surface_form_initial_hidden = []
# #         lengths = []
# #
# #         # go through each unique surface form
# #         for s_id, surface_form in enumerate(surface_forms_sorted_by_memory_length):
# #
# #             # print(surface_form)
# #
# #             # get embedding history of this surface form and bring to flair.device
# #             surface_form_embedding_history = surface_form_history[surface_form]
# #             for i in range(len(surface_form_embedding_history)):
# #                 all_surface_form_tensor[i, s_id] = surface_form_embedding_history[i]
# #
# #             lengths.append(len(surface_form_embedding_history))
# #
# #             # initialize first hidden state if necessary
# #             if surface_form not in self.state_history:
# #                 self.state_history[surface_form] = self.initial_hidden if self.rnn_type == 'GRU' else [self.lstm_init_h, self.lstm_init_c]
# #                 # print(self.state_history[surface_form][:10])
# #
# #             all_surface_form_initial_hidden.append(
# #                 self.state_history[surface_form]
# #             )
# #
# #         # print('initial hidden:')
# #
# #         # make batch tensors
# #         all_surface_form_histories = all_surface_form_tensor
# #
# #         # dropout!
# #         all_surface_form_histories = self.dropout(all_surface_form_histories)
# #
# #         # reproject if set
# #         if self.reproject:
# #             all_surface_form_histories = self.reprojection_layer(
# #                 all_surface_form_histories
# #             )
# #
# #         # print([initial_hidden[0] for initial_hidden in all_surface_form_initial_hidden])
# #
# #         # print(all_surface_form_initial_hidden)
# #
# #         # get initial hidden state of each surface form
# #         if self.rnn_type == 'GRU':
# #             all_surface_form_initial_hidden = torch.stack(
# #                 all_surface_form_initial_hidden, 0
# #             ).unsqueeze(0)
# #
# #         # print(all_surface_form_initial_hidden)
# #
# #         if self.rnn_type == 'LSTM':
# #
# #             all_surface_form_h = torch.stack(
# #                 [initial_hidden[0] for initial_hidden in all_surface_form_initial_hidden], 0
# #             ).unsqueeze(0)
# #             all_surface_form_c = torch.stack(
# #                 [initial_hidden[1] for initial_hidden in all_surface_form_initial_hidden], 0
# #             ).unsqueeze(0)
# #             all_surface_form_initial_hidden = [all_surface_form_h, all_surface_form_c]
# #
# #         # send through RNN
# #         packed = torch.nn.utils.rnn.pack_padded_sequence(all_surface_form_histories, lengths)
# #
# #         # print('surface form initial hidden:')
# #         # print(all_surface_form_initial_hidden)
# #
# #         packed_output, hidden = self.rnn(packed, all_surface_form_initial_hidden)
# #
# #         rnn_out, hidden_2 = torch.nn.utils.rnn.pad_packed_sequence(
# #             packed_output
# #         )
# #
# #         # print('hidden:')
# #         # print(hidden)
# #         # print('rnn_out:')
# #         # print(rnn_out)
# #
# #         # hidden = hidden
# #
# #         # go through each unique surface form and update state history
# #         for idx, surface_form in enumerate(surface_forms_sorted_by_memory_length):
# #             # retrieve hidden states for this surface form, and the last initial hidden state
# #
# #             # set as new state history of surface form (before: .cpu())
# #             if self.rnn_type == 'GRU':
# #                 last_hidden_state_of_token = hidden.detach()[0:lengths[idx], idx].squeeze()
# #             if self.rnn_type == 'LSTM':
# #                 last_hidden_state_of_token = (hidden[0].detach()[0:lengths[idx], idx].squeeze(), hidden[1].detach()[0:lengths[idx], idx].squeeze())
# #
# #             # print(surface_form)
# #             # print(last_hidden_state_of_token)
# #
# #             # print('last hidden state:')
# #             # print(last_hidden_state_of_token)
# #             self.state_history[surface_form] = last_hidden_state_of_token
# #
# #             # asd
# #
# #         counter_of_handled_surface_forms = Counter()
# #         # finally, go through each token of each sentence and set the embedding
# #
# #         for sentence in sentences:
# #
# #             for token in sentence:
# #
# #                 idx = surface_forms_sorted_by_memory_length.index(token.text)
# #
# #                 # get index of present token in memory
# #                 memory_index = lengths[idx] - counter_of_surface_forms_in_batch[token.text] + \
# #                                counter_of_handled_surface_forms[token.text]
# #
# #                 # get embedding
# #                 embedding = rnn_out[memory_index, idx]
# #
# #                 if self.use_batch_memory_state:
# #                     # memory_index = lengths[idx] - 1
# #
# #                     # batch_embedding = rnn_out[memory_index, idx]
# #                     # print('batch+hidden:')
# #                     # print(batch_embedding)
# #                     if self.rnn_type == 'GRU':
# #                         last_hidden_state_of_token = hidden[0:lengths[idx], idx].squeeze()
# #                     if self.rnn_type == 'LSTM':
# #                         last_hidden_state_of_token = torch.cat([hidden[0][0:lengths[idx], idx].squeeze(),
# #                                                       hidden[1][0:lengths[idx], idx].squeeze()])
# #                     # print(last_hidden_state_of_token)
# #
# #                     # embedding = batch_embedding
# #                     embedding = torch.cat([embedding, last_hidden_state_of_token])
# #
# #                 token.set_embedding(self.name, embedding)
# #
# #                 if self.training:
# #                     flip = float(random.randint(0, 99) + 1) / 100
# #                     if flip < self.word_dropout:
# #                         for subembedding in self.sub_embedding_names:
# #                             token._embeddings[subembedding] = \
# #                                 torch.zeros(token._embeddings[subembedding].size(),
# #                                             device=token._embeddings[subembedding].device)
# #
# #                 if not self.concat_word_embeddings:
# #                     for subembedding in self.sub_embedding_names:
# #                         if subembedding in token._embeddings.keys():
# #                             del token._embeddings[subembedding]
# #
# #                 counter_of_handled_surface_forms[token.text] += 1
# #
# #         return sentences
# #
# #     @property
# #     def embedding_length(self) -> int:
# #         return self.__embedding_length
# #
# #     def extra_repr(self):
# #         return f"[use_batch_memory_state='{self.use_batch_memory_state}', concat_word_embeddings='{self.concat_word_embeddings}']"
#
#
#
#
# class MemoryEmbeddingsRNN(TokenEmbeddings):
#     def __init__(
#             self,
#             contextual_embeddings: TokenEmbeddings,
#             concat_word_embeddings: bool = True,
#             hidden_states: int = 64,
#             reproject: bool = True,
#             max_memory_length: int = 8,
#             dropout: float = 0.5,
#             word_dropout: float = 0.0,
#             local_dropout: float = 0.1,
#             use_batch_memory_state: bool = False,
#             rnn_type: str = 'GRU',
#             train_initial_hidden_state: bool = True,
#     ):
#
#         super().__init__()
#
#         # variables
#         self.hidden_states = hidden_states
#         self.max_memory_length: int = max_memory_length
#         self.effective_memory_length: int = self.max_memory_length
#         self.concat_word_embeddings: bool = concat_word_embeddings
#         self.word_dropout: float = word_dropout
#         self.static_embeddings: bool = False
#
#         self.context_embeddings: TokenEmbeddings = contextual_embeddings
#
#         self.sub_embedding_names = (
#             [emb.name for emb in self.context_embeddings.embeddings]
#             if type(self.context_embeddings) is StackedEmbeddings
#             else [self.context_embeddings.name]
#         )
#
#         state_length = self.hidden_states if not use_batch_memory_state else self.hidden_states * 2
#
#         # determine embedding length
#         self.__embedding_length = (
#             self.context_embeddings.embedding_length + state_length
#             if self.concat_word_embeddings
#             else 0 + state_length
#         )
#
#         # the memory
#         self.word_history = {}
#         self.state_history = {}
#
#         # the NN
#         self.dropout = torch.nn.Dropout(dropout)
#         self.local_dropout = torch.nn.Dropout(local_dropout)
#
#         self.reproject: bool = reproject
#         if self.reproject:
#             self.reprojection_layer = torch.nn.Linear(
#                 self.context_embeddings.embedding_length,
#                 self.context_embeddings.embedding_length,
#             )
#
#         self.rnn_type = rnn_type
#
#         if self.rnn_type == 'GRU':
#             self.rnn = torch.nn.GRU(
#                 self.context_embeddings.embedding_length,
#                 self.hidden_states,
#                 num_layers=1,
#                 bidirectional=False,
#             )
#             # default zero-state tensor
#             self.initial_hidden = torch.zeros(self.hidden_states, device=flair.device)
#
#             # make initial hidden state trainable if set
#             # if train_initial_hidden_state:
#             #     self.initial_hidden = torch.nn.Parameter(
#             #         self.initial_hidden,
#             #         requires_grad=True,
#             #     )
#
#         # if self.rnn_type == 'LSTM':
#         #     self.rnn = torch.nn.LSTM(
#         #         self.context_embeddings.embedding_length,
#         #         self.hidden_states,
#         #         num_layers=1,
#         #         bidirectional=False,
#         #     )
#         #     # default zero-state tensor
#         #     self.initial_hidden = (
#         #         torch.zeros(self.hidden_states, device=flair.device),
#         #         torch.zeros(self.hidden_states, device=flair.device)
#         #     )
#         #
#         #     # make initial hidden state trainable if set
#         #     if train_initial_hidden_state:
#         #         self.initial_hidden = [
#         #             torch.nn.Parameter(torch.zeros(self.hidden_states, device=flair.device),
#         #                                requires_grad=True, ),
#         #             torch.nn.Parameter(torch.zeros(self.hidden_states, device=flair.device),
#         #                                requires_grad=True, )]
#
#         self.name = self.context_embeddings.name + "-memory"
#
#         self.use_batch_memory_state = use_batch_memory_state
#
#         self.to(flair.device)
#
#         self.eval()
#
#     def train(self, mode=True):
#         super().train(mode=mode)
#         compress = True
#         if mode:
#             print("train mode resetting embeddings")
#             self.word_history =  {}
#             self.state_history = {}
#             self.effective_memory_length = self.max_memory_length
#
#         elif compress:
#             # memory is wiped each time we do evaluation
#             print("prediction mode no backprop")
#
#             self.word_history = {}
#             for word in self.state_history:
#                 self.state_history[word] = [self.state_history[word][-1].clone()]
#             self.effective_memory_length = 1
#
#     def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
#
#         self.context_embeddings.embed(sentences)
#
#         # determine and add to history of each token
#         counter_of_surface_forms_in_batch = Counter()
#         surface_form_history = {}
#         for sentence in sentences:
#             for token in sentence:
#                 counter_of_surface_forms_in_batch[token.text] += 1
#
#                 # detach previous gradients otherwise all hell breaks loose
#                 if counter_of_surface_forms_in_batch[token.text] == 1 and token.text in self.word_history:
#                     for idx, tensor in enumerate(self.word_history[token.text]):
#                         tensor = tensor.detach()
#                         self.word_history[token.text][idx] = tensor
#
#                 # add current embedding to the memory
#                 local_embedding = token.get_subembedding(
#                     self.sub_embedding_names
#                 )
#
#                 if token.text not in self.word_history:
#                     self.word_history[token.text] = [local_embedding]
#                 else:
#                     self.word_history[token.text].append(local_embedding)
#
#                 surface_form_history[token.text] = self.word_history[token.text]
#
#         # sort surface forms by longest history
#         surface_forms_sorted_by_memory_length = sorted(surface_form_history,
#                                                        key=lambda k: len(surface_form_history[k]),
#                                                        reverse=True)
#         logest_memory_length_in_batch = len(surface_form_history[surface_forms_sorted_by_memory_length[0]])
#
#         # initialize zero-padded word embeddings tensor
#         all_surface_form_tensor = torch.zeros(
#             [
#                 logest_memory_length_in_batch,
#                 len(surface_form_history),
#                 self.context_embeddings.embedding_length,
#             ],
#             dtype=torch.float,
#             device=flair.device,
#         )
#
#         all_surface_form_initial_hidden = []
#         lengths = []
#
#         # go through each unique surface form
#         for s_id, surface_form in enumerate(surface_forms_sorted_by_memory_length):
#
#             # print(surface_form)
#
#             # get embedding history of this surface form and bring to flair.device
#             surface_form_embedding_history = surface_form_history[surface_form]
#             length_of_surface_form_embedding_history = len(surface_form_embedding_history)
#
#             # set all but current state
#             for i in range(length_of_surface_form_embedding_history - 1):
#                 all_surface_form_tensor[i, s_id] = surface_form_embedding_history[i]
#
#             # set current state
#             all_surface_form_tensor[length_of_surface_form_embedding_history - 1, s_id] = \
#                 self.local_dropout(surface_form_embedding_history[
#                     length_of_surface_form_embedding_history - 1])
#
#             lengths.append(length_of_surface_form_embedding_history)
#
#             # truncate surface form history if necessary
#             window = 0 if len(self.word_history[surface_form]) < self.effective_memory_length else len(
#                 self.word_history[surface_form]) - self.effective_memory_length
#             # if window > 0:
#             self.word_history[surface_form] = self.word_history[surface_form][
#                                                   window:self.effective_memory_length + window]
#
#             # initialize first hidden state if necessary
#             if surface_form not in self.state_history:
#                 self.state_history[surface_form] = [self.initial_hidden]
#
#             all_surface_form_initial_hidden.append(
#                 self.state_history[surface_form][0]
#             )
#
#         # make batch tensors
#         all_surface_form_histories = all_surface_form_tensor
#
#         # dropout!
#         all_surface_form_histories = self.dropout(all_surface_form_histories)
#
#         # reproject if set
#         if self.reproject:
#             all_surface_form_histories = self.reprojection_layer(
#                 all_surface_form_histories
#             )
#
#         # get initial hidden state of each surface form
#         if self.rnn_type == 'GRU':
#             all_surface_form_initial_hidden = torch.stack(
#                 all_surface_form_initial_hidden, 0
#             ).unsqueeze(0)
#
#         # if self.rnn_type == 'LSTM':
#         #     all_surface_form_h = torch.stack(
#         #         [initial_hidden[0] for initial_hidden in all_surface_form_initial_hidden], 0
#         #     ).unsqueeze(0)
#         #     all_surface_form_c = torch.stack(
#         #         [initial_hidden[1] for initial_hidden in all_surface_form_initial_hidden], 0
#         #     ).unsqueeze(0)
#         #     all_surface_form_initial_hidden = [all_surface_form_h, all_surface_form_c]
#
#         # send through RNN
#         packed = torch.nn.utils.rnn.pack_padded_sequence(all_surface_form_histories, lengths)
#
#         packed_output, hidden = self.rnn(packed, all_surface_form_initial_hidden)
#
#         rnn_out, hidden = torch.nn.utils.rnn.pad_packed_sequence(
#             packed_output
#         )
#
#         # go through each unique surface form and update state history
#         for idx, surface_form in enumerate(surface_forms_sorted_by_memory_length):
#             # retrieve hidden states for this surface form, and the last initial hidden state
#             hidden_states_of_surface_form = rnn_out[0:lengths[idx], idx]
#             last_initial_hidden_state = all_surface_form_initial_hidden[0][idx].unsqueeze(0)
#
#             # concat both to get new hidden state history
#             hidden_state_history = torch.cat([last_initial_hidden_state, hidden_states_of_surface_form], dim=0)
#
#             # if history is too long, truncate to window
#             window = 0 if hidden_state_history.size(0) < self.effective_memory_length else hidden_state_history.size(
#                 0) - self.effective_memory_length
#             state_history_window = hidden_state_history.detach()[window:self.effective_memory_length + window]
#
#             # set as new state history of surface form (before: .cpu())
#             self.state_history[surface_form] = state_history_window
#
#         counter_of_handled_surface_forms = Counter()
#
#         # finally, go through each token of each sentence and set the embedding
#         for sentence in sentences:
#
#             for token in sentence:
#
#                 idx = surface_forms_sorted_by_memory_length.index(token.text)
#
#                 # get index of present token in memory
#                 memory_index = lengths[idx] - counter_of_surface_forms_in_batch[token.text] + \
#                                counter_of_handled_surface_forms[token.text]
#
#                 # get embedding
#                 embedding = rnn_out[memory_index, idx]
#
#                 if self.use_batch_memory_state:
#                     memory_index = lengths[idx] - 1
#                     batch_embedding = rnn_out[memory_index, idx]
#                     # embedding = batch_embedding
#                     embedding = torch.cat([embedding, batch_embedding])
#
#                 token.set_embedding(self.name, embedding)
#
#                 # if self.training:
#                 #     flip = float(random.randint(0, 99) + 1) / 100
#                 #     if flip < self.word_dropout:
#                 #         for subembedding in self.sub_embedding_names:
#                 #             token._embeddings[subembedding] = \
#                 #                 torch.zeros(token._embeddings[subembedding].size(),
#                 #                             device=token._embeddings[subembedding].device)
#
#                 if not self.concat_word_embeddings:
#                     for subembedding in self.sub_embedding_names:
#                         if subembedding in token._embeddings.keys():
#                             del token._embeddings[subembedding]
#
#                 counter_of_handled_surface_forms[token.text] += 1
#
#         return sentences
#
#     @property
#     def embedding_length(self) -> int:
#         return self.__embedding_length
#
#     def extra_repr(self):
#         return f"[use_batch_memory_state='{self.use_batch_memory_state}', concat_word_embeddings='{self.concat_word_embeddings}']"
#
#
# class ContextualizedWordEmbeddings(TokenEmbeddings):
#     def __init__(
#             self,
#             hidden_size: int,
#             embeddings: TokenEmbeddings,
#             rnn_layers: int = 1,
#             dropout: float = 0.0,
#             word_dropout: float = 0.05,
#             locked_dropout: float = 0.5,
#     ):
#         super().__init__()
#
#         self.hidden_size = hidden_size
#         self.__embedding_length = hidden_size * 2
#         self.rnn_layers: int = rnn_layers
#         self.name = embeddings.name + '-contextualized'
#         self.static_embeddings = False
#
#         self.trained_epochs: int = 0
#
#         self.embeddings = embeddings
#
#         # initialize the network architecture
#         self.nlayers: int = rnn_layers
#         self.hidden_word = None
#
#         # dropouts
#         self.use_dropout: float = dropout
#         self.use_word_dropout: float = word_dropout
#         self.use_locked_dropout: float = locked_dropout
#
#         if dropout > 0.0:
#             self.dropout = torch.nn.Dropout(dropout)
#
#         if word_dropout > 0.0:
#             self.word_dropout = flair.nn.WordDropout(word_dropout)
#
#         if locked_dropout > 0.0:
#             self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
#
#         rnn_input_dim: int = self.embeddings.embedding_length
#
#         self.relearn_embeddings: bool = True
#
#         if self.relearn_embeddings:
#             self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)
#
#         self.bidirectional = True
#         self.rnn_type = "LSTM"
#
#         if self.rnn_type in ["LSTM", "GRU"]:
#
#             self.rnn = getattr(torch.nn, self.rnn_type)(
#                 rnn_input_dim,
#                 hidden_size,
#                 num_layers=self.nlayers,
#                 dropout=0.0 if self.nlayers == 1 else 0.5,
#                 bidirectional=True,
#             )
#
#         self.to(flair.device)
#
#         self.eval()
#
#     def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
#
#         self.embeddings.embed(sentences)
#
#         lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
#         longest_token_sequence_in_batch: int = max(lengths)
#
#         # initialize zero-padded word embeddings tensor
#         sentence_tensor = torch.zeros(
#             [
#                 len(sentences),
#                 longest_token_sequence_in_batch,
#                 self.embeddings.embedding_length,
#             ],
#             dtype=torch.float,
#             device=flair.device,
#         )
#
#         for s_id, sentence in enumerate(sentences):
#             # fill values with word embeddings
#             sentence_tensor[s_id][: len(sentence)] = torch.cat(
#                 [token.get_embedding().unsqueeze(0) for token in sentence], 0
#             )
#
#         # TODO: this can only be removed once the implementations of word_dropout and locked_dropout have a batch_first mode
#         sentence_tensor = sentence_tensor.transpose_(0, 1)
#
#         # --------------------------------------------------------------------
#         # FF PART
#         # --------------------------------------------------------------------
#         if self.use_dropout > 0.0:
#             sentence_tensor = self.dropout(sentence_tensor)
#         if self.use_word_dropout > 0.0:
#             sentence_tensor = self.word_dropout(sentence_tensor)
#         if self.use_locked_dropout > 0.0:
#             sentence_tensor = self.locked_dropout(sentence_tensor)
#
#         if self.relearn_embeddings:
#             sentence_tensor = self.embedding2nn(sentence_tensor)
#
#         packed = torch.nn.utils.rnn.pack_padded_sequence(
#             sentence_tensor, lengths, enforce_sorted=False
#         )
#
#         # if initial hidden state is trainable, use this state
#         rnn_output, hidden = self.rnn(packed)
#
#         sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
#             rnn_output, batch_first=True
#         )
#
#         if self.use_dropout > 0.0:
#             sentence_tensor = self.dropout(sentence_tensor)
#         if self.use_locked_dropout > 0.0:
#             sentence_tensor = self.locked_dropout(sentence_tensor)
#
#         # print(sentence_tensor.size())
#         for s_id, sentence in enumerate(sentences):
#             for t_id, token in enumerate(sentence):
#                 # print(sentence_tensor[s_id, t_id].size())
#                 token.clear_embeddings()
#                 token.set_embedding(self.name, sentence_tensor[s_id, t_id])
#
#         return sentences
#
#     @property
#     def embedding_length(self) -> int:
#         return self.__embedding_length
#
# class ContextualizedMemoryEmbeddings(TokenEmbeddings):
#     def __init__(
#             self,
#             hidden_size: int,
#             embeddings: TokenEmbeddings,
#             rnn_layers: int = 1,
#             memory_hidden_size: int = 128,
#             dropout: float = 0.0,
#             word_dropout: float = 0.0,
#             locked_dropout: float = 0.5,
#             history_dropout: float = 0.1,
#             local_dropout: float = 0.1,
#             max_memory_length = 8,
#             concat_word_embeddings = False,
#     ):
#         super().__init__()
#
#         self.concat_word_embeddings = concat_word_embeddings
#         self.max_memory_length: int = max_memory_length
#         self.effective_memory_length: int = self.max_memory_length
#
#         self.hidden_size = hidden_size
#
#         self.__embedding_length = memory_hidden_size
#         if concat_word_embeddings:
#             self.__embedding_length += hidden_size * 2
#
#         self.rnn_layers: int = rnn_layers
#         self.name = embeddings.name + '-contextualized'
#         self.static_embeddings = False
#
#         self.trained_epochs: int = 0
#
#         self.embeddings = embeddings
#
#         # initialize the network architecture
#         self.nlayers: int = rnn_layers
#         self.hidden_word = None
#
#         # dropouts
#         self.use_dropout: float = dropout
#         self.use_word_dropout: float = word_dropout
#         self.use_locked_dropout: float = locked_dropout
#
#         if dropout > 0.0:
#             self.dropout = torch.nn.Dropout(dropout)
#
#         if word_dropout > 0.0:
#             self.word_dropout = flair.nn.WordDropout(word_dropout)
#
#         if locked_dropout > 0.0:
#             self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
#
#         rnn_input_dim: int = self.embeddings.embedding_length
#
#         self.relearn_embeddings: bool = True
#
#         if self.relearn_embeddings:
#             self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)
#
#         self.bidirectional = True
#         self.rnn_type = "LSTM"
#
#         if self.rnn_type in ["LSTM", "GRU"]:
#             self.rnn = getattr(torch.nn, self.rnn_type)(
#                 rnn_input_dim,
#                 hidden_size,
#                 num_layers=self.nlayers,
#                 dropout=0.0 if self.nlayers == 1 else 0.5,
#                 bidirectional=True,
#             )
#
#         # the memory
#         self.word_history = {}
#         self.state_history = {}
#
#         # the NN
#         self.history_dropout = torch.nn.Dropout(history_dropout)
#         self.local_dropout = torch.nn.Dropout(local_dropout)
#
#         self.memory_rnn = torch.nn.GRU(
#             hidden_size * 2,
#             memory_hidden_size,
#             num_layers=1,
#             bidirectional=False,
#         )
#         # default zero-state tensor
#         self.initial_hidden = torch.zeros(memory_hidden_size, device=flair.device)
#
#         self.to(flair.device)
#
#         self.eval()
#
#     def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
#
#         self.embeddings.embed(sentences)
#
#         lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
#         longest_token_sequence_in_batch: int = max(lengths)
#
#         # initialize zero-padded word embeddings tensor
#         sentence_tensor = torch.zeros(
#             [
#                 len(sentences),
#                 longest_token_sequence_in_batch,
#                 self.embeddings.embedding_length,
#             ],
#             dtype=torch.float,
#             device=flair.device,
#         )
#
#         for s_id, sentence in enumerate(sentences):
#             # fill values with word embeddings
#             sentence_tensor[s_id][: len(sentence)] = torch.cat(
#                 [token.get_embedding().unsqueeze(0) for token in sentence], 0
#             )
#
#         # TODO: this can only be removed once the implementations of word_dropout and locked_dropout have a batch_first mode
#         sentence_tensor = sentence_tensor.transpose_(0, 1)
#
#         # --------------------------------------------------------------------
#         # FF PART
#         # --------------------------------------------------------------------
#         if self.use_dropout > 0.0:
#             sentence_tensor = self.dropout(sentence_tensor)
#         if self.use_word_dropout > 0.0:
#             sentence_tensor = self.word_dropout(sentence_tensor)
#         if self.use_locked_dropout > 0.0:
#             sentence_tensor = self.locked_dropout(sentence_tensor)
#
#         if self.relearn_embeddings:
#             sentence_tensor = self.embedding2nn(sentence_tensor)
#
#         packed = torch.nn.utils.rnn.pack_padded_sequence(
#             sentence_tensor, lengths, enforce_sorted=False
#         )
#
#         # if initial hidden state is trainable, use this state
#         rnn_output, hidden = self.rnn(packed)
#
#         sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
#             rnn_output, batch_first=True
#         )
#
#         # determine and add to history of each token
#         counter_of_surface_forms_in_batch = Counter()
#         surface_form_history = {}
#         for s_id, sentence in enumerate(sentences):
#             for t_id, token in enumerate(sentence):
#                 counter_of_surface_forms_in_batch[token.text] += 1
#
#                 # detach previous gradients otherwise all hell breaks loose
#                 if counter_of_surface_forms_in_batch[token.text] == 1 and token.text in self.word_history:
#                     for idx, tensor in enumerate(self.word_history[token.text]):
#                         tensor = tensor.detach()
#                         self.word_history[token.text][idx] = tensor
#
#                 # add current embedding to the memory
#                 local_embedding = sentence_tensor[s_id, t_id]
#
#                 if token.text not in self.word_history:
#                     self.word_history[token.text] = [local_embedding]
#                 else:
#                     self.word_history[token.text].append(local_embedding)
#
#                 surface_form_history[token.text] = self.word_history[token.text]
#
#         # sort surface forms by longest history
#         surface_forms_sorted_by_memory_length = sorted(surface_form_history,
#                                                        key=lambda k: len(surface_form_history[k]),
#                                                        reverse=True)
#         logest_memory_length_in_batch = len(surface_form_history[surface_forms_sorted_by_memory_length[0]])
#
#         # initialize zero-padded word embeddings tensor
#         all_surface_form_tensor = torch.zeros(
#             [
#                 logest_memory_length_in_batch,
#                 len(surface_form_history),
#                 self.hidden_size * 2,
#             ],
#             dtype=torch.float,
#             device=flair.device,
#         )
#
#         all_surface_form_initial_hidden = []
#         lengths = []
#
#         # go through each unique surface form
#         for s_id, surface_form in enumerate(surface_forms_sorted_by_memory_length):
#
#             # print(surface_form)
#
#             # get embedding history of this surface form and bring to flair.device
#             surface_form_embedding_history = surface_form_history[surface_form]
#             length_of_surface_form_embedding_history = len(surface_form_embedding_history)
#
#             # set all but current state
#             for i in range(length_of_surface_form_embedding_history - 1):
#                 all_surface_form_tensor[i, s_id] = surface_form_embedding_history[i]
#
#             # set current state
#             all_surface_form_tensor[length_of_surface_form_embedding_history - 1, s_id] = \
#                 self.local_dropout(surface_form_embedding_history[
#                                        length_of_surface_form_embedding_history - 1])
#
#             lengths.append(length_of_surface_form_embedding_history)
#
#             # truncate surface form history if necessary
#             window = 0 if len(self.word_history[surface_form]) < self.effective_memory_length else len(
#                 self.word_history[surface_form]) - self.effective_memory_length
#             # if window > 0:
#             self.word_history[surface_form] = self.word_history[surface_form][
#                                               window:self.effective_memory_length + window]
#
#             # initialize first hidden state if necessary
#             if surface_form not in self.state_history:
#                 self.state_history[surface_form] = [self.initial_hidden]
#
#             all_surface_form_initial_hidden.append(
#                 self.state_history[surface_form][0]
#             )
#
#         # make batch tensors
#         all_surface_form_histories = all_surface_form_tensor
#
#         # dropout!
#         all_surface_form_histories = self.history_dropout(all_surface_form_histories)
#
#         # reproject if set
#         # if self.reproject:
#         #     all_surface_form_histories = self.reprojection_layer(
#         #         all_surface_form_histories
#         #     )
#
#         # get initial hidden state of each surface form
#         all_surface_form_initial_hidden = torch.stack(
#             all_surface_form_initial_hidden, 0
#         ).unsqueeze(0)
#
#         # send through RNN
#         packed = torch.nn.utils.rnn.pack_padded_sequence(all_surface_form_histories, lengths)
#
#         packed_output, hidden = self.memory_rnn(packed, all_surface_form_initial_hidden)
#
#         rnn_out, hidden = torch.nn.utils.rnn.pad_packed_sequence(
#             packed_output
#         )
#
#         # go through each unique surface form and update state history
#         for idx, surface_form in enumerate(surface_forms_sorted_by_memory_length):
#             # retrieve hidden states for this surface form, and the last initial hidden state
#             hidden_states_of_surface_form = rnn_out[0:lengths[idx], idx]
#             last_initial_hidden_state = all_surface_form_initial_hidden[0][idx].unsqueeze(0)
#
#             # concat both to get new hidden state history
#             hidden_state_history = torch.cat([last_initial_hidden_state, hidden_states_of_surface_form],
#                                              dim=0)
#
#             # if history is too long, truncate to window
#             window = 0 if hidden_state_history.size(
#                 0) < self.effective_memory_length else hidden_state_history.size(
#                 0) - self.effective_memory_length
#             state_history_window = hidden_state_history.detach()[
#                                    window:self.effective_memory_length + window]
#
#             # set as new state history of surface form (before: .cpu())
#             self.state_history[surface_form] = state_history_window
#
#         counter_of_handled_surface_forms = Counter()
#
#         # finally, go through each token of each sentence and set the embedding
#         for s_id, sentence in enumerate(sentences):
#             for t_id, token in enumerate(sentence):
#
#                 idx = surface_forms_sorted_by_memory_length.index(token.text)
#
#                 # get index of present token in memory
#                 memory_index = lengths[idx] - counter_of_surface_forms_in_batch[token.text] + \
#                                counter_of_handled_surface_forms[token.text]
#
#                 # get embedding
#                 embedding = rnn_out[memory_index, idx]
#
#                 # if self.use_batch_memory_state:
#                 #     memory_index = lengths[idx] - 1
#                 #     batch_embedding = rnn_out[memory_index, idx]
#                 #     # embedding = batch_embedding
#                 #     embedding = torch.cat([embedding, batch_embedding])
#
#                 token.clear_embeddings()
#                 if self.concat_word_embeddings:
#                     embedding = torch.cat([embedding, sentence_tensor[s_id, t_id]])
#
#                 token.set_embedding(self.name, embedding)
#
#                 # if not self.concat_word_embeddings:
#                 #     for subembedding in self.sub_embedding_names:
#                 #         if subembedding in token._embeddings.keys():
#                 #             del token._embeddings[subembedding]
#
#                 counter_of_handled_surface_forms[token.text] += 1
#
#
#         return sentences
#
#     @property
#     def embedding_length(self) -> int:
#         return self.__embedding_length
#
#     def train(self, mode=True):
#         super().train(mode=mode)
#         compress = True
#         if mode:
#             print("train mode resetting embeddings")
#             self.word_history =  {}
#             self.state_history = {}
#             self.effective_memory_length = self.max_memory_length
#
#         elif compress:
#             # memory is wiped each time we do evaluation
#             print("prediction mode no backprop")
#
#             self.word_history = {}
#             for word in self.state_history:
#                 self.state_history[word] = [self.state_history[word][-1].clone()]
#             self.effective_memory_length = 1
#
