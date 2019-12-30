from typing import List, Tuple

import torch
from torch.nn.modules.rnn import LSTM

import flair
from flair.data import Sentence

from flair.embeddings import TokenEmbeddings
from flair.nn import MemoryCell


class MemoryEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            base_embeddings: TokenEmbeddings,
            context_hidden_states: int = 256,
            memory_hidden_states: int = 32,
            max_memory_length: int = 16,
            dropout: float = 0.0,
            locked_dropout: float = 0.5,
            word_dropout: float = 0.05,
            use_batch_memory_state: bool = False,
            train_initial_hidden_state: bool = True,
            reflective_memory: bool = True,
    ):

        super().__init__()

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        # reprojection layer
        self.relearn_embeddings: bool = True
        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(base_embeddings.embedding_length,
                                                base_embeddings.embedding_length)

        self.context_rnn = LSTM(
            base_embeddings.embedding_length,
            context_hidden_states,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.memory_cell = MemoryCell(
            input_size=context_hidden_states * 2,
            concat_word_embeddings=True if reflective_memory else False,
            hidden_states=memory_hidden_states,
            max_memory_length=max_memory_length,
            dropout=dropout,
            locked_dropout=locked_dropout,
            word_dropout=word_dropout,
            use_batch_memory_state=use_batch_memory_state,
            train_initial=train_initial_hidden_state,
            reflective_memory=reflective_memory,
        )

        # variables
        self.static_embeddings: bool = False

        self.base_embeddings: TokenEmbeddings = base_embeddings

        # determine embedding length
        self.__embedding_length = memory_hidden_states + (context_hidden_states * 2) if reflective_memory else memory_hidden_states
        if use_batch_memory_state:
            self.__embedding_length += memory_hidden_states

        self.name = self.base_embeddings.name + "-memory"

        self.to(flair.device)

        self.eval()

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for sentence in sentences:
            sentence.clear_embeddings([self.name])

        # embed with base word embeddings
        self.base_embeddings.embed(sentences)

        # make a sentence tensor
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.base_embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        t_ids: List[List[str]] = []
        for s_id, sentence in enumerate(sentences):
            t_ids.append([token.text for token in sentence])

            all_embs = list()

            for index_token, token in enumerate(sentence):
                embs = token.get_each_embedding()
                if not all_embs:
                    all_embs = [list() for _ in range(len(embs))]
                for index_emb, emb in enumerate(embs):
                    all_embs[index_emb].append(emb)

            concat_word_emb = [torch.stack(embs) for embs in all_embs]
            concat_sentence_emb = torch.cat(concat_word_emb, dim=1)
            sentence_tensor[s_id][: len(sentence)] = concat_sentence_emb

        # drop and reproject
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        # context RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )
        rnn_output, hidden = self.context_rnn(packed)
        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_output, batch_first=True
        )

        # send through memory cell
        sentence_tensor = self.memory_cell(sentence_tensor, t_ids)

        # finally, go through each token of each sentence and set the embedding
        for s_id, sentence in enumerate(sentences):

            for t_id, token in enumerate(sentence):
                token._embeddings = {}
                token.set_embedding(self.name, sentence_tensor[s_id, t_id].clone())

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length


class MemoryCellEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            contextual_embeddings: TokenEmbeddings,
            hidden_states: int = 32,
            max_memory_length: int = 16,
            dropout: float = 0.5,
            locked_dropout: float = 0.5,
            use_batch_memory_state: bool = False,
            train_initial_hidden_state: bool = True,
            memory_only: bool = True,
            memento_mode: bool = False,
    ):

        super().__init__()

        self.max_memory_length = max_memory_length

        self.memory_cell = MemoryCell(
            input_size=contextual_embeddings.embedding_length,
            concat_word_embeddings=False,
            hidden_states=hidden_states,
            max_memory_length=max_memory_length,
            dropout=dropout,
            locked_dropout=locked_dropout,
            use_batch_memory_state=use_batch_memory_state,
            train_initial=train_initial_hidden_state,
            reflective_memory=False,
            memento_mode = memento_mode
        )

        # variables
        self.static_embeddings: bool = False
        self.memory_only: bool = memory_only

        self.context_embeddings: TokenEmbeddings = contextual_embeddings

        # determine embedding length
        self.__embedding_length = hidden_states
        if not self.memory_only:
            self.__embedding_length += contextual_embeddings.embedding_length

        self.name = self.context_embeddings.name + "-memory"

        self.to(flair.device)

        self.eval()

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for sentence in sentences:
            sentence.clear_embeddings([self.name])

        self.context_embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.context_embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        t_ids: List[List[str]] = []
        for s_id, sentence in enumerate(sentences):
            # fill values with word embeddings
            sentence_tensor[s_id][: len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0
            )
            t_ids.append([token.text for token in sentence])

        sentence_tensor = self.memory_cell(sentence_tensor, t_ids)

        # finally, go through each token of each sentence and set the embedding
        for s_id, sentence in enumerate(sentences):

            for t_id, token in enumerate(sentence):
                if self.memory_only:
                    token._embeddings = {}
                token.set_embedding(self.name, sentence_tensor[s_id, t_id].clone())

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    # def extra_repr(self):
    #     return f"[use_batch_memory_state='{self.use_batch_memory_state}', concat_word_embeddings='{self.concat_word_embeddings}']"
#
# # class HistoryEmbeddings(TokenEmbeddings):
# #     def __init__(
# #             self,
# #             contextual_embeddings: TokenEmbeddings,
# #             concat_word_embeddings: bool = True,
# #             hidden_states: int = 64,
# #             reproject: bool = True,
# #             max_memory_length: int = 8,
# #             dropout: float = 0.5,
# #             word_dropout: float = 0.1,
# #             use_batch_memory_state: bool = False,
# #             rnn_type: str = 'GRU',
# #             train_initial_hidden_state: bool = True,
# #     ):
# #
# #         super().__init__()
# #
# #         # variables
# #         self.hidden_states = hidden_states
# #         self.max_memory_length: int = max_memory_length
# #         self.effective_memory_length: int = self.max_memory_length
# #         self.concat_word_embeddings: bool = concat_word_embeddings
# #         self.static_embeddings: bool = False
# #         self.word_dropout = word_dropout
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
# #         # determine embedding length
# #         self.__embedding_length = (
# #             self.context_embeddings.embedding_length + state_length
# #             if self.concat_word_embeddings
# #             else 0 + state_length
# #         )
# #
# #         # the memory
# #         self.word_history = {}
# #         self.state_history = {}
# #
# #         # the NN
# #         self.dropout = torch.nn.Dropout(dropout)
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
# #         # default zero-state embedding
# #         self.empty_word_tensor = torch.zeros(self.context_embeddings.embedding_length, device=flair.device)
# #
# #         self.name = self.context_embeddings.name + "-memory"
# #
# #         self.use_batch_memory_state = use_batch_memory_state
# #
# #         self.to(flair.device)
# #
# #         self.eval()
# #
# #     def train(self, mode=True):
# #         super().train(mode=mode)
# #         compress = True
# #         if mode:
# #             print("train mode resetting embeddings")
# #             self.word_history = {}
# #             self.state_history = {}
# #             self.effective_memory_length = self.max_memory_length
# #
# #         elif compress:
# #             # memory is wiped each time we do evaluation
# #             print("prediction mode no backprop")
# #             self.effective_memory_length = 1
# #
# #     def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
# #
# #         self.context_embeddings.embed(sentences)
# #
# #         # if 'Blargl' in self.word_history:
# #         #     print(f'Blargl word history: {len(self.word_history["Blargl"])} {self.word_history["Blargl"]}')
# #         #     print(f'Blargl state history: {self.state_history["Blargl"]}')
# #
# #         # determine and add to history of each token
# #         counter_of_surface_forms_in_batch = Counter()
# #         surface_form_history = {}
# #         for sentence in sentences:
# #             for token in sentence:
# #                 counter_of_surface_forms_in_batch[token.text] += 1
# #
# #                 # detach previous gradients otherwise all hell breaks loose
# #                 if counter_of_surface_forms_in_batch[token.text] == 1 and token.text in self.word_history:
# #                     for idx, tensor in enumerate(self.word_history[token.text]):
# #                         tensor = tensor.detach()
# #                         self.word_history[token.text][idx] = tensor
# #
# #                 # add current embedding to the memory
# #                 local_embedding = token.get_subembedding(
# #                     self.sub_embedding_names
# #                 )
# #
# #                 # empty tensor represents word we've never seen before
# #                 if token.text not in self.word_history:
# #                     self.word_history[token.text] = [self.empty_word_tensor]
# #
# #                 self.word_history[token.text].append(local_embedding)
# #
# #                 surface_form_history[token.text] = self.word_history[token.text]
# #
# #         # sort surface forms by longest history
# #         surface_forms_sorted_by_memory_length = sorted(surface_form_history,
# #                                                        key=lambda k: len(surface_form_history[k]),
# #                                                        reverse=True)
# #         longest_memory_length_in_batch = len(surface_form_history[surface_forms_sorted_by_memory_length[0]])
# #
# #         # print(longest_memory_length_in_batch)
# #
# #         # initialize zero-padded word embeddings tensor
# #         all_surface_form_tensor = torch.zeros(
# #             [
# #                 longest_memory_length_in_batch,
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
# #             # print(surface_form_history[surface_form])
# #
# #             # get EMBEDDING HISTORY of this surface form and bring to flair.device
# #             surface_form_embedding_history = surface_form_history[surface_form]
# #             length_of_surface_form_embedding_history = len(surface_form_embedding_history)
# #
# #             # set all states
# #             for i in range(length_of_surface_form_embedding_history):
# #                 all_surface_form_tensor[i, s_id] = surface_form_embedding_history[i]
# #
# #             # the current state does not count as length
# #             lengths.append(length_of_surface_form_embedding_history)
# #
# #             # get STATE HISTORY of this surface form
# #
# #             # initialize first hidden state if necessary
# #             if surface_form not in self.state_history:
# #                 self.state_history[surface_form] = self.initial_hidden
# #
# #             # print(f'initial hidden state of {surface_form}')
# #             # print(self.state_history[surface_form])
# #
# #             all_surface_form_initial_hidden.append(
# #                 self.state_history[surface_form]
# #             )
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
# #         # get initial hidden state of each surface form
# #         if self.rnn_type == 'GRU':
# #             all_surface_form_initial_hidden = torch.stack(
# #                 all_surface_form_initial_hidden, 0
# #             ).unsqueeze(0)
# #
# #         # print(all_surface_form_histories.size())
# #
# #         # send through RNN
# #         packed = torch.nn.utils.rnn.pack_padded_sequence(all_surface_form_histories, lengths)
# #
# #         packed_output, hidden = self.rnn(packed, all_surface_form_initial_hidden)
# #
# #         rnn_out, hidden = torch.nn.utils.rnn.pad_packed_sequence(
# #             packed_output
# #         )
# #
# #         # print('rnn out:')
# #         # print(rnn_out)
# #
# #         # go through each unique surface form and update word and state history
# #         for idx, surface_form in enumerate(surface_forms_sorted_by_memory_length):
# #
# #             # truncate surface form history if necessary
# #             window = 0 if len(self.word_history[surface_form]) < self.effective_memory_length else len(
# #                 self.word_history[surface_form]) - self.effective_memory_length
# #
# #             if window > 0:
# #                 # print(f'truncating {surface_form} from {self.word_history[surface_form]}')
# #                 self.word_history[surface_form] = self.word_history[surface_form][
# #                                                   window:self.effective_memory_length + window]
# #                 # print(f'to {self.word_history[surface_form]}')
# #                 # print(window)
# #
# #                 # print(rnn_out[0:lengths[idx], idx])
# #                 self.state_history[surface_form] = rnn_out[window - 1, idx].detach()
# #
# #         counter_of_handled_surface_forms = Counter()
# #
# #         # finally, go through each token of each sentence and set the embedding
# #         for sentence in sentences:
# #
# #             for token in sentence:
# #
# #                 idx = surface_forms_sorted_by_memory_length.index(token.text)
# #
# #                 # get index of present token in memory
# #                 memory_index = lengths[idx] - counter_of_surface_forms_in_batch[token.text] + \
# #                                counter_of_handled_surface_forms[token.text] - 1
# #
# #                 # get embedding
# #                 embedding = rnn_out[memory_index, idx]
# #
# #                 if self.use_batch_memory_state:
# #                     memory_index = lengths[idx] - 1
# #                     batch_embedding = rnn_out[memory_index, idx]
# #                     # embedding = batch_embedding
# #                     embedding = torch.cat([embedding, batch_embedding])
# #
# #                 token.set_embedding(self.name, embedding)
# #
# #                 if not self.concat_word_embeddings:
# #                     for subembedding in self.sub_embedding_names:
# #                         if subembedding in token._embeddings.keys():
# #                             del token._embeddings[subembedding]
# #
# #                 elif self.training:
# #                     flip = float(random.randint(0, 99) + 1) / 100
# #                     if flip < self.word_dropout:
# #                         for subembedding in self.sub_embedding_names:
# #                             token._embeddings[subembedding] = \
# #                                 torch.zeros(token._embeddings[subembedding].size(),
# #                                             device=token._embeddings[subembedding].device)
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
