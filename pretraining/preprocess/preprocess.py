import os
import hydra
import multiprocess as mp

from tqdm import tqdm
from fastcore.transform import Pipeline
import numpy as np
import networkx
import networkx.algorithms.centrality as centrality
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from stanza.server import CoreNLPClient, StartServer
from rouge_score import rouge_scorer

from src.module.vocabulary import Vocab
from src.utils import read_jsonl, write_jsonl, get_ori_path


class Tokenize:
    def __init__(self, max_src_nsents):
        self.max_src_nsents = max_src_nsents

    def __call__(self, data):
        if data is None:
            return data

        src_txt, tgt_txt = data["src"], data["tgt"]

        src_result = corenlp.annotate(src_txt)
        tgt_result = corenlp.annotate(tgt_txt)

        src_tokens = [
            [token.word.lower() for token in sentence.token]
            for sentence in src_result.sentence
        ]
        tgt_tokens = [
            [token.word.lower() for token in sentence.token]
            for sentence in tgt_result.sentence
        ]

        src_txt = [" ".join(subtokens).strip() for subtokens in src_tokens]
        tgt_txt = [" ".join(subtokens).strip() for subtokens in tgt_tokens]

        src_txt = src_txt[: self.max_src_nsents]

        return {"src_txt": src_txt, "tgt_txt": tgt_txt}


class CalculateOracle:
    def __init__(self, n_sent_of_summary):
        self.n_sent_of_summary = n_sent_of_summary
        self.scorer = rouge_scorer.RougeScorer(rouge_types=["rouge1", "rouge2"])

    def _generate_oracle(self, src_sents, gold_summary):
        selected = []
        best_rouge = 0.0

        for _ in range(self.n_sent_of_summary):
            cur_max_rouge = 0.0
            cur_max_idx = -1
            for i in range(len(src_sents)):
                if i not in selected:
                    candi = "\n".join(
                        [src_sents[idx] for idx in sorted(selected + [i])]
                    )
                    score = self.scorer.score(candi, gold_summary)
                    mean_score = []
                    for rouge_type, rouge_score in score.items():
                        mean_score.append(rouge_score.fmeasure)
                    cur_rouge = float(np.sum(mean_score))
                    if cur_rouge > cur_max_rouge:
                        cur_max_rouge = cur_rouge
                        cur_max_idx = i
            if cur_max_rouge != 0.0 and cur_max_rouge > best_rouge:
                selected.append(cur_max_idx)
                best_rouge = cur_max_rouge
            else:
                break

        labels = [0 if i not in selected else 1 for i in range(len(src_sents))]

        return labels

    def __call__(self, data):
        if data is None:
            return None

        src_txt = data["src_txt"]
        tgt_txt = data["tgt_txt"]

        labels = self._generate_oracle(src_txt, "\n".join(tgt_txt))

        return {
            "src_txt": data["src_txt"],
            "tgt_txt": data["tgt_txt"],
            "labels": labels,
        }


def create_vocab(dataset, vocab_fp):
    allwords = []
    for data in dataset:
        src = " ".join(data["src_txt"])
        tgt = " ".join(data["tgt_txt"])

        allwords.extend(src.split())
        allwords.extend(tgt.split())

    temp_vocab = nltk.FreqDist(allwords)
    fout = open(vocab_fp, "w")
    for k, v in temp_vocab.most_common():
        fout.write("%s\t%d\n" % (k, v))
    fout.close()


def create_filterwords(dataset, vocab, nwords_low_tfidf):
    filterwords = stopwords.words("english")
    filterwords.extend([",", ".", ":", ";", "?", "(", ")", "[",  "]", "&", "!", "*", "@", "#"])
    filterwords.extend(["$", "%", "''",  "'", "`", "``", "-", "--", "|", "\/", "'s", "'m", "'re", "n't", "'d",])

    # filter words with low tfidf scores
    # nwords_low_tfidf means number of words filtered by tfidf scores
    docs = []
    for data in dataset:
        docs.append(" ".join(data["src_txt"]))
    vectorizer = CountVectorizer(lowercase=True)
    wordcount = vectorizer.fit_transform(docs)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)

    selected_ids = np.argsort(np.array(tfidf_matrix.mean(0))[0])
    del tfidf_matrix
    words = vectorizer.get_feature_names()
    for i in selected_ids[:nwords_low_tfidf]:
        w = words[i]
        if vocab.stoi(w) == vocab.stoi("[UNK]"):
            continue
        filterwords.extend([w])

    filterids = [vocab.stoi(w.lower()) for w in filterwords]
    filterids.append(vocab.stoi("[PAD]"))

    filter_mask = np.ones(vocab.size())
    filter_mask[filterids] = 0.0

    return filter_mask, filterwords


class ConstructGraph:
    def __init__(self, vocab, filter_mask, max_src_sent_len):
        self.vocab = vocab
        self.filter_mask = filter_mask
        self.max_src_sent_len = max_src_sent_len

    def _pad_sent(self, seq, pad_id):
        max_len = self.max_src_sent_len
        if len(seq) > max_len:
            seq = seq[:max_len]
        if len(seq) < max_len:
            seq.extend([pad_id] * (max_len - len(seq)))
        return seq

    def __call__(self, data):
        if data is None:
            return None

        src_txt = data["src_txt"]

        vocab_ids_per_sent = []
        doc_vocab_ids = []
        for sent in src_txt:
            words = sent.split()
            temp = [self.vocab.stoi(w) for w in words]
            doc_vocab_ids.extend(temp)
            vocab_ids_per_sent.append(self._pad_sent(temp, self.vocab.stoi("[PAD]")))

        doc_vocab_ids = np.array(doc_vocab_ids)
        vocab_ids_per_sent = np.array(vocab_ids_per_sent)
        all_vocab = np.zeros(self.vocab.size())
        all_vocab[doc_vocab_ids.reshape(-1)] = 1.0
        all_vocab = all_vocab * self.filter_mask
        # the value is id in vocab, and indice is node id in graph
        ntov = np.argwhere(all_vocab == 1.0).reshape(-1)
        n_word_nodes = ntov.size
        vton = np.zeros(self.vocab.size()).astype(np.int64)
        vton[ntov] = np.arange(n_word_nodes)
        # the value is sentence id in the order of the document, and indice is node id in graph
        ntos = np.arange(len(src_txt)) + n_word_nodes
        n_sent_nodes = len(ntos)

        # Adjacency matrix, shape: [n_sent_node, vocab_size]
        adj = np.zeros((n_sent_nodes, self.vocab.size()))
        adj[np.arange(n_sent_nodes).reshape(-1, 1), vocab_ids_per_sent] = 1.0
        # filter stopwords, punctuation, and words with low tfidf socres.
        adj = adj * self.filter_mask.reshape(1, -1)
        # edges shape: [n_edges, 2]
        edges = np.argwhere(adj == 1.0)
        # transform vocab id to node id
        edges[:, 1] = vton[edges[:, 1]]
        # transform sentence id to node id
        edges[:, 0] = edges[:, 0] + n_word_nodes

        return {
            "src_txt": data["src_txt"],
            "tgt_txt": data["tgt_txt"],
            "labels": data["labels"],
            "word_node_ids": ntov,
            "sent_node_ids": vocab_ids_per_sent,
            "edges": edges,
        }


class CalculateSemanticEdge:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, data):
        if data is None:
            return data

        src_txt = data["src_txt"]
        word_node_ids = data["word_node_ids"]
        sent_node_ids = data["sent_node_ids"]
        n_word_nodes = len(word_node_ids)
        n_sent_nodes = len(sent_node_ids)
        edges = data["edges"]

        wordcount = np.zeros((n_sent_nodes, n_word_nodes)).astype(np.int64)
        vton = np.zeros(self.vocab.size()).astype(np.int64)
        vton[word_node_ids] = np.arange(n_word_nodes)
        ind = (
            np.arange(n_sent_nodes).reshape(-1, 1),
            vton[sent_node_ids],
        )
        np.add.at(wordcount, ind, 1)

        pad_id = self.vocab.stoi("[PAD]")
        wordcount[:, pad_id] = 0

        tf_idf_transformer = TfidfTransformer()
        tfidf_matrix = tf_idf_transformer.fit_transform(wordcount).toarray()

        semantic_edge = tfidf_matrix[edges[:, 0] - n_word_nodes, edges[:, 1]]

        return {
            "src_txt": src_txt,
            "tgt_txt": data["tgt_txt"],
            "labels": data["labels"],
            "word_node_ids": word_node_ids,
            "sent_node_ids": sent_node_ids,
            "edges": edges,
            "semantic_edge": semantic_edge,
        }


class CalculateTopologyEdge:
    def __init__(self):
        pass

    def __call__(self, data):
        if data is None:
            return None

        word_node_ids = data["word_node_ids"]
        sent_node_ids = data["sent_node_ids"]
        n_word_nodes = len(word_node_ids)
        n_sent_nodes = len(sent_node_ids)
        edges = data["edges"]

        g = networkx.Graph()
        g.add_nodes_from([i for i in range(n_word_nodes + n_sent_nodes)])
        g.add_edges_from(edges)

        topology_edge = [v for k, v in centrality.edge_load_centrality(g).items()]

        word_node_centrality = [[] for _ in range(n_word_nodes)]
        sent_node_centrality = [[] for _ in range(n_sent_nodes)]

        for centrality_func in (
            centrality.degree_centrality,
            centrality.eigenvector_centrality_numpy,
            centrality.closeness_centrality,
            centrality.betweenness_centrality,
        ):
            cent = centrality_func(g)
            for k, v in cent.items():
                if k < n_word_nodes:
                    word_node_centrality[k].append(v)
                else:
                    sent_node_centrality[k - n_word_nodes].append(v)

        return {
            "src_txt": data["src_txt"],
            "tgt_txt": data["tgt_txt"],
            "labels": data["labels"],
            "word_node_ids": word_node_ids.tolist(),
            "sent_node_ids": sent_node_ids.tolist(),
            "word_node_centrality": word_node_centrality,
            "sent_node_centrality": sent_node_centrality,
            "edges": edges.tolist(),
            "semantic_edge": data["semantic_edge"].tolist(),
            "topology_edge": topology_edge,
        }


@hydra.main(config_path="configuration", config_name="preprocess")
def preprocess(hps):
    #logger = MyLogger.build_from_hydra("preprocess")

    os.makedirs(os.path.dirname(get_ori_path(hps.target_path)), exist_ok=True)
    os.makedirs(os.path.dirname(get_ori_path(hps.vocab_path)), exist_ok=True)

    #logger.info("Reading input dataset...")
    dataset = read_jsonl(get_ori_path(hps.input_path))

    global corenlp
    corenlp = CoreNLPClient(
        start_server=StartServer.TRY_START,
        endpoint="http://localhost:9999",
        annotators=["tokenize", "ssplit"],
        timeout=3000,
        memory="10G",
        be_quiet=True,
    )
    tokenize = Tokenize(hps.max_src_nsents)
    caloracle = CalculateOracle(hps.n_sent_of_summary)
    pre_pipeline = Pipeline([tokenize, caloracle])

    #logger.info("Tokenize and calculate oracle...")
    pbar = tqdm(total=len(dataset), desc="pre_pineline")
    with mp.Pool(hps.n_proc) as pool:
        result = pool.imap(pre_pipeline, dataset)
        mdataset = []
        for r in result:
            if r:
                mdataset.append(r)
            pbar.update()
    pbar.close()

    #logger.info("Create vocabulary...")
    if "train" in hps.target_path:
        create_vocab(mdataset, get_ori_path(hps.vocab_path))
    vocab = Vocab(get_ori_path(hps.vocab_path), max_size=hps.max_vocab_size)

    fileter_mask, filterwords = create_filterwords(
        mdataset, vocab, hps.nwords_low_tfidf
    )

    graph_constructer = ConstructGraph(vocab, fileter_mask, hps.max_src_sent_len)
    cal_semantic_edge = CalculateSemanticEdge(vocab)
    cal_topology_edge = CalculateTopologyEdge()
    post_pipeline = Pipeline([graph_constructer, cal_semantic_edge, cal_topology_edge])

    #logger.info("Construct graph...")
    pbar = tqdm(total=len(mdataset), desc="post_pipeline")
    with mp.Pool(hps.n_proc) as pool:
        result = pool.imap(post_pipeline, mdataset)
        fdataset = []
        for r in result:
            if r:
                fdataset.append(r)
            pbar.update()
    pbar.close()

    #logger.info("Writing results...")
    write_jsonl(get_ori_path(hps.target_path), fdataset)


if __name__ == "__main__":
    preprocess()
    """
    try:
        preprocess()
    finally:
        corenlp.stop()
        corenlp.atexit_kill()
    """
