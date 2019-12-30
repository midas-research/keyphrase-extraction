import logging
import random
from abc import abstractmethod
from pathlib import Path
from typing import List, Union

from torch.nn import CosineSimilarity
from torch.utils.data import Dataset, random_split

import flair, torch
from flair.data import Sentence, Token
from flair.datasets import FlairDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Result
import random

from flair.data import Sentence, Corpus
from typing import Dict, List, Tuple


class BiSentence:
    def __init__(self, source: Sentence, target: Sentence, alignments: str = None):
        self.source: Sentence = source
        self.target: Sentence = target
        self.alignments = alignments

    def get_aligned_tokens(self):
        aligned_tokens = []
        # print(f"'{self.alignments}'")
        for alignment in self.alignments.split(' '):
            indices = alignment.split('-')
            aligned_tokens.append((self.source[int(indices[0])], self.target[int(indices[1])]))

        return aligned_tokens

    def __repr__(self):
        return f'BiSentence: \n - source: {self.source}\n - target: {self.target}\n'

    def __str__(self):
        return f'BiSentence: \n - source: {self.source}\n - target: {self.target}\n'

    def clear_embeddings(self, also_clear_word_embeddings: bool = True):
        pass


log = logging.getLogger("flair")


class BiTextCorpus(Corpus):
    def __init__(
            self,
            path_to_bitext: Union[str, Path],
            path_to_alignments: Union[str, Path],
            in_memory: bool = False,
            max_lines: int = None,
    ):
        """
        Instantiates a Corpus from text classification-formatted task data

        :param path_to_bitext: base folder with the task data
        :return: a Corpus with annotated train, dev and test data
        """

        if type(path_to_bitext) == str:
            path_to_bitext: Path = Path(path_to_bitext)
        if type(path_to_alignments) == str:
            path_to_alignments: Path = Path(path_to_alignments)

        log.info("Reading data from {}".format(path_to_bitext))

        train: Dataset = BiTextDataset(
            path_to_bitext,
            path_to_alignments,
            max_lines,
            in_memory=in_memory,
        )

        train_length = len(train)
        dev_size: int = round(train_length / 10)
        splits = random_split(train, [train_length - dev_size, dev_size])
        train = splits[0]
        test = splits[1]

        train_length = len(train)
        dev_size: int = round(train_length / 10)
        splits = random_split(train, [train_length - dev_size, dev_size])
        train = splits[0]
        dev = splits[1]

        super(BiTextCorpus, self).__init__(
            train, dev, test, name=path_to_bitext
        )


class BiTextDataset(FlairDataset):
    def __init__(
            self,
            path_to_bitext: Union[str, Path],
            path_to_alignments: Union[str, Path],
            max_lines: int = 100000,
            in_memory: bool = False,
    ):

        if type(path_to_bitext) == str:
            path_to_bitext: Path = Path(path_to_bitext)
        if type(path_to_alignments) == str:
            path_to_alignments: Path = Path(path_to_alignments)

        assert path_to_bitext.exists()
        assert path_to_alignments.exists()

        self.in_memory = in_memory

        if self.in_memory:
            self.biSentences: List[BiSentence] = []
        else:
            self.lines: List[Tuple[str, str]] = []

        self.total_sentence_count: int = 0

        self.path_to_file = path_to_bitext
        self.path_to_alignments = path_to_alignments

        with open(str(path_to_bitext), 'r') as f, open(str(path_to_alignments)) as a:
            for line, word_alignments in zip(f, a):

                if word_alignments.strip() == '': continue

                self.total_sentence_count += 1
                if self.total_sentence_count % 10000 == 0:
                    print(self.total_sentence_count)

                if self.in_memory:
                    biSentence = self._parse_line_to_biSentence(line, word_alignments)
                    self.biSentences.append(biSentence)
                else:
                    self.lines.append((line, word_alignments))

                if self.total_sentence_count >= max_lines:
                    break


    def _parse_line_to_biSentence(
            self, line: str, word_alignments: str
    ):
        source_sentence: Sentence = Sentence(line.split('|||')[0].strip())
        target_sentence: Sentence = Sentence(line.split('|||')[1].strip())
        sentence_pair: BiSentence = BiSentence(source_sentence, target_sentence, word_alignments.strip())

        return sentence_pair

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> BiSentence:
        if self.in_memory:
            return self.biSentences[index]
        else:
            return self._parse_line_to_biSentence(self.lines[index][0], self.lines[index][1])


# class ParallelCorpus(Corpus):
#
#     def __init__(self, file, alignments, max_lines: int = 1000000, in_memory=False):
#
#         # get the training data
#         train: List[BiSentence] = []
#
#         # get the test data
#         test: List[BiSentence] = []
#
#         # get the test data
#         dev: List[BiSentence] = []
#
#         count: int = 0
#         total_count: int = 0
#
#         with open(file, 'r') as f, open(alignments) as a:
#             for line, word_alignments in zip(f, a):
#
#                 total_count += 1
#                 if total_count % 10000 == 0:
#                     print(total_count)
#
#                 if word_alignments.strip() == '': continue
#
#                 source_sentence: Sentence = Sentence(line.split('|||')[0].strip())
#                 target_sentence: Sentence = Sentence(line.split('|||')[1].strip())
#                 sentence_pair: BiSentence = BiSentence(source_sentence, target_sentence, word_alignments.strip())
#
#                 count += 1
#
#                 if count == 9:
#                     test.append(sentence_pair)
#                 elif count == 10:
#                     dev.append(sentence_pair)
#                     count = 0
#                 else:
#                     train.append(sentence_pair)
#
#                 if total_count >= max_lines:
#                     break
#
#         super(ParallelCorpus, self).__init__(SentenceDataset(train), SentenceDataset(dev), SentenceDataset(test))


class TripletRankLoss(torch.nn.Module):
    def __init__(self, margin=0.1, distance="cosine", n_modalities=2):
        # TODO: direction weights, so one can be more interested in one direction of alignment than the other,
        #  or if one trusts more labels in one direction than in the other
        super(TripletRankLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        if self.distance not in ["cosine", "sqL2"]:
            raise Exception("Only cosine distance implemented")
        self.n_modalities = n_modalities

        self.relu = torch.nn.ReLU()

        if self.n_modalities != 2:
            raise Exception("Only implemented for two sources/modalities")

    def forward(self, inputs):
        n_data, in_dim = inputs.shape
        assert in_dim % self.n_modalities == 0
        aligned_modality_dim = in_dim // self.n_modalities
        off_diag_ones = torch.ones(n_data, device=flair.device) - torch.eye(
            n_data, device=flair.device
        )
        modality_representations_list = [
            inputs[
            :,
            modality_id
            * aligned_modality_dim: (modality_id + 1)
                                    * aligned_modality_dim,
            ]
            for modality_id in range(self.n_modalities)
        ]
        if self.distance == "cosine":
            modality_representations_list_normalized = [
                modality_representation
                / modality_representation.norm(dim=1, keepdim=True)
                for modality_representation in modality_representations_list
            ]
            dotprod_matrix = torch.matmul(
                modality_representations_list_normalized[0],
                modality_representations_list_normalized[1].t(),
            )
            loss_matrix = self.relu(
                self.margin * off_diag_ones
                + dotprod_matrix
                - torch.diag(dotprod_matrix).view(n_data, 1)
            )
        elif self.distance == "sqL2":
            dotprod_matrix = torch.matmul(
                modality_representations_list[0], modality_representations_list[1].t()
            )
            cross_term_norm_matrix = (
                    modality_representations_list[0].sum(dim=1, keepdim=True)
                    + modality_representations_list[1].sum(dim=1, keepdim=True).t()
            )
            distance_matrix = cross_term_norm_matrix - 2 * dotprod_matrix
            loss_matrix = self.relu(
                torch.diag(distance_matrix).view(n_data, 1)
                - distance_matrix
                - self.margin * off_diag_ones
            )
        loss = torch.sum(off_diag_ones * loss_matrix) / (n_data * (n_data - 1))

        return loss


class MultilingualEmbeddings(TokenEmbeddings):
    def __init__(
            self, embeddings: Union[str, TokenEmbeddings], map_type: str = "linear", embedding_length = None,
    ):
        super().__init__()

        self.static_embeddings: bool = False

        self.map_type = map_type

        self.dropout = torch.nn.Dropout(0.1)

        if isinstance(embeddings, TokenEmbeddings):
            self.embeddings: TokenEmbeddings = embeddings

            if embedding_length is None:
                embedding_length = embeddings.embedding_length

            self.map = torch.nn.Linear(
                embeddings.embedding_length, embedding_length
            )
            self.init_weights(self.map)

            if map_type == "nonlinear":
                self.relu = torch.nn.ReLU()
                self.map_in = torch.nn.Linear(
                    embedding_length, embedding_length
                )
                self.init_weights(self.map_in)

            self.name: str = self.embeddings.name + "-mapped"

        else:
            print('loading')
            loaded_mapper = WordEmbeddingMapper.load(embeddings).embeddings
            self.embeddings = loaded_mapper.embeddings
            embedding_length = self.embeddings.embedding_length
            self.map = loaded_mapper.map
            if loaded_mapper.map_type == "nonlinear":
                self.relu = loaded_mapper.relu
                self.map_in = loaded_mapper.map_in
            self.name: str = loaded_mapper.name
            # hacky ...
            self.static_embeddings = True
            self.dropout = torch.nn.Dropout(0.)

        self.base_embeddings: List[str] = [self.embeddings.name] if self.embeddings.name != 'Stack' else [
            stack_embedding.name for stack_embedding in self.embeddings.embeddings]

        self.__embedding_length: int = embedding_length

        self.eval()

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.eye_(m.weight)
            m.bias.data.fill_(0.0)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        self.embeddings.embed(sentences)

        all_embeddings = [
            token.get_subembedding(self.base_embeddings).unsqueeze(0)
            for sentence in sentences
            for token in sentence
        ]

        embeddings_tensor = torch.cat(all_embeddings, dim=0).to(flair.device)

        embeddings_tensor = self.dropout(embeddings_tensor)

        mapped_tensor = self.map(embeddings_tensor)

        if self.map_type == "nonlinear":
            mapped_tensor = self.relu(mapped_tensor)
            mapped_tensor = self.map_in(mapped_tensor)

        i = 0
        for sentence in sentences:
            for token in sentence:
                token.clear_embeddings()
                embedding = mapped_tensor[i, :]
                if self.static_embeddings:
                    embedding = embedding.detach()
                token.set_embedding(self.name, embedding)
                i += 1

        return sentences

    # def __str__(self):
    #     return self.name


class WordEmbeddingMapper(flair.nn.Model):
    def __init__(self, embeddings: TokenEmbeddings, map_type: str = "linear", embedding_length = None,):

        super(WordEmbeddingMapper, self).__init__()

        self.embeddings = MultilingualEmbeddings(embeddings, map_type, embedding_length)

        self.loss_function = TripletRankLoss()

        self.to(flair.device)

    def forward_loss(self, data_points: Union[List[Sentence], Sentence]) -> torch.tensor:

        # first add embeddings
        source_sentences: List[Sentence] = []
        target_sentences: List[Sentence] = []

        for sentence_pair in data_points:
            source_sentences.append(sentence_pair.source)
            target_sentences.append(sentence_pair.target)

        self.embeddings.embed(source_sentences)
        self.embeddings.embed(target_sentences)

        all_source_embeddings = []
        all_target_embeddings = []

        parallel_surface_forms = []
        source_surface_forms = []
        target_surface_forms = []
        for sentence_pair in data_points:
            for word_alignment in sentence_pair.get_aligned_tokens():
                # if word_alignment[0].text in source_surface_forms:
                #     continue
                # if word_alignment[1].text in target_surface_forms:
                #     continue
                sf = f"{word_alignment[0].text}~{word_alignment[1].text}"
                if sf in parallel_surface_forms:
                    continue
                parallel_surface_forms.append(sf)
                # source_surface_forms.append(word_alignment[0].text)
                # target_surface_forms.append(word_alignment[1].text)
                # print(word_alignment)

                all_source_embeddings.append(
                    word_alignment[0]
                        .get_subembedding([self.embeddings.name])
                        .unsqueeze(0)
                )
                all_target_embeddings.append(
                    word_alignment[1]
                        .get_subembedding([self.embeddings.name])
                        .unsqueeze(0)
                )

        # print(source_surface_forms)
        # print(len(source_surface_forms))
        # print(target_surface_forms)
        # print(len(target_surface_forms))
        # asd

        if len(all_source_embeddings) == 0 or len(all_target_embeddings) == 0:
            print(data_points)
            return 0.0

        all_source_embeddings = torch.cat(all_source_embeddings).to(flair.device)

        all_target_embeddings = torch.cat(all_target_embeddings).to(flair.device)

        input = torch.cat([all_source_embeddings, all_target_embeddings], dim=1)
        loss = self.loss_function.forward(input)

        return loss

    def predict(
            self, data_points: Union[List[Sentence], Sentence], mini_batch_size=32
    ) -> List[Sentence]:
        pass

    def evaluate(
            self,
            sentences: List[BiSentence],
            eval_mini_batch_size: int = 32,
            embeddings_in_memory: bool = False,
            out_path: Path = None,
            num_workers: int = 8,
    ) -> (Result, float):

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            batch_loader = DataLoader(
                sentences,
                batch_size=eval_mini_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            for batch in batch_loader:
                batch_no += 1

                with torch.no_grad():
                    loss = self.forward_loss(batch)

                eval_loss += loss

                clear_embeddings(
                    batch, also_clear_word_embeddings=not embeddings_in_memory
                )

            if batch_no > 0:
                eval_loss /= batch_no

            result = Result(
                main_score=1.0 - eval_loss,
                log_line=f"{1. - eval_loss}",
                log_header="LOSS",
                detailed_results=1.0 - eval_loss,
            )

            return result, eval_loss

    def _get_state_dict(self):

        model_state = {
            "state_dict": self.state_dict(),
            "base_embeddings": self.embeddings.embeddings,
            "map_type": self.embeddings.map_type,
            "embedding_length": self.embeddings.embedding_length,
        }
        return model_state

    def _init_model_with_state_dict(state):
        length_ = state["embedding_length"] if "embedding_length" in state else None
        model = WordEmbeddingMapper(embeddings=state["base_embeddings"], map_type=state["map_type"], embedding_length=length_)
        model.load_state_dict(state["state_dict"])
        return model

    def _fetch_model(model_name) -> str:
        return model_name
