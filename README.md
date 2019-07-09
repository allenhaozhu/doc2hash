# Doc2hash: Learning Discrete Latent variables for Documents Retrieval (NAACL 2019)
This work is an implementation of [Doc2hash: Learning Discrete Latent variables for Documents Retrieval](https://www.aclweb.org/anthology/N19-1232) and based on the codes of VDSH (Variational Deep Semantic Hashing (SIGIR'2017) (http://students.engr.scu.edu/~schaidar/paper/Variational_Deep_Hashing_for_Text_Documents.pdf)).

# Platform
This project uses python 3.6 and pytorch 0.4

# Available dataset
- reuters, rcv1, ng20, tmc, dbpedia, agnews, yahooanswer

# Prepare dataset
We provide a script to generate the datasets in the preprocess folder. You need to download the raw datasets for TMC. 

# Training and Evaluating the model
To train the unsupervised learning model, run the following command:
```
python train_VDSH.py -d [dataset name] -g [gpu number] -b [number of bits]
```

To train the supervised learning model, run the following command:
```
python train_VDSH_S.py -d [dataset name] -g [gpu number] -b [number of bits]
```
OR
```
python train_VDSH_SP.py -d [dataset name] -g [gpu number] -b [number of bits]
```

# Bibtex
```
@inproceedings{zhang-zhu-2019-doc2hash,
    title = "{D}oc2hash: Learning Discrete Latent variables for Documents Retrieval",
    author = "Zhang, Yifei  and
      Zhu, Hao",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1232",
    pages = "2235--2240",
    abstract = "Learning to hash via generative model has become a powerful paradigm for fast similarity search in documents retrieval. To get binary representation (i.e., hash codes), the discrete distribution prior (i.e., Bernoulli Distribution) is applied to train the variational autoencoder (VAE). However, the discrete stochastic layer is usually incompatible with the backpropagation in the training stage, and thus causes a gradient flow problem because of non-differentiable operators. The reparameterization trick of sampling from a discrete distribution usually inc non-differentiable operators. In this paper, we propose a method, Doc2hash, that solves the gradient flow problem of the discrete stochastic layer by using continuous relaxation on priors, and trains the generative model in an end-to-end manner to generate hash codes. In qualitative and quantitative experiments, we show the proposed model outperforms other state-of-art methods.",
}

@inproceedings{Chaidaroon:2017:VDS:3077136.3080816,
 author = {Chaidaroon, Suthee and Fang, Yi},
 title = {Variational Deep Semantic Hashing for Text Documents},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR '17},
 year = {2017},
 isbn = {978-1-4503-5022-8},
 location = {Shinjuku, Tokyo, Japan},
 pages = {75--84},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3077136.3080816},
 doi = {10.1145/3077136.3080816},
 acmid = {3080816},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deep learning, semantic hashing, variational autoencoder},
}
```

# doc2hash
