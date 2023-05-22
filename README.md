# Long Term Memory in AI - Vector Search and Databases
COS 495 is given at Princeton during the Fall semester 2023 by

* [Edo Liberty](https://scholar.google.com/citations?user=QHS_pZAAAAAJ&hl=en), the Founder and CEO of [Pinecone](www.pinecone.io), the world's leading Vector Database.

* [Matthijs Douze](https://scholar.google.com/citations?user=0eFZtREAAAAJ&hl=en]) the creator and lead maintainer of [FAISS](https://github.com/facebookresearch/faiss) the most popular and advanced open source library for vector search.

The course covers the core concepts, algorithms, and data structures used for modern vector search systems and platforms. An advanced undergraduate or graduate student with some hands-on experience in linear algebra, probability, algorithms, and data structures should be able to follow this course.


## Abstract
Long Term Memory is a fundamental capability in the modern AI Stack. At their core, these systems are using Vector search. Vector search is also a fundamental tool for systems that manipulate large collections of media like search engines, knowledge bases, content moderation tools, recommendation systems, etc. As such, the discipline lays at the intersection of Artificial Intelligence and Database Management Systems. This course will cover the scientific foundations and practical implementation of vector search applications, algorithms, and systems. The course will be evaluated with project and in-class presentation. 


## Syllabus

**The class contents below are tentative.**

1. Introduction to Vector Search 
	* Embeddings as an information bottleneck. Instead of learning end-to-end, use embeddings as an intermediate representation
	* Advantages: scalability, instant updates, and explainability
	* Typical volumes of data and scalability. Embeddings are the only way to manage / access large databases
	* The embedding contract: the embedding extractor and embedding indexer agree on the meaning of the distance. Separation of concerns.
	* The vector space model in information retrieval
	* Vector embeddings in machine learning
	
1. Text embeddings
	* 2-layer word embeddings. Word2vec and fastText, obtained via a factorization of a co-occurrence matrix. Embedding arithmetic: king + woman - man = queen, (already based on similarity search)
	* Sentence embeddings: How to train, masked LM. Properties of sentence embeddings.
	* Large Language Models: reasoning as an emerging property of a LM. What happens when the training set = the whole web

1. Image embeddings 
	* Pixel structures of images. Early works on direct pixel indexing
	* Traditional CV models. Global descriptors (GIST). Local descriptors (SIFT and friends)Direct indexing of local descriptors for image matching, local descriptor pooling (Fisher, VLAD)
	* Convolutional Neural Nets. Off-the-shelf models. Trained specifically (contrastive learning, self-supervised learning)
	* Modern Computer Vision models 

	
1. Low Dimensional Vector Search 
	* Exact vecrtor search
	* k-d tree, space partitioning based algorithms, proof, structures, and asymptotic behavior
	* Probabilistic inequalities. Recap of basic inequalities: Markov, Chernoof, Hoeffding
	* Concentration Of Measure phenomena. Orthogonality of random vectors
	* Curse of dimensionality. The failure of space partitioning

1. Dimensionality Reduction
	* Principal Components Analysis, optimal dimension reduction in the sum of squared distance measure
	* Fast PCA algorithms 
	* Random Projections. Gaussian random i.i.d. dimension reduction 
	* Fast Random Projections. Accelerating the above to near linear time

1. Locality Sensitive Hashing
	* Definition of Approximate Nearest Neighbor Search (ANNS)
	* Criteria: Speed / accuracy / memory usage / updateability / index construction time 
	* Definition of Locality Sensitive Hashing and examples
	* The LSH Algorithm
	* LSH Analysis, proof of correctness, and asymptotics
	
1. Clustering
	* Semantic clustering: properties (purity, as aid for annotation)
	* Clustering from a similarity graph (spectral clustering, agglomerative clustering)
	* Vector clustering: mean squared error criterion. Tradeoff with number of clusters
	* Relationship between vector clustering and quantization (OOD extension) 
	* The k-means clustering measure and Lloyd's algorithm
	* Lloyd's optimality conditions
	* Initialization strategies (kmeans++, progressive dimensions with PCA)
	* The inverted file model. Relationship with sparse matrices
	
1. Quantization for lossy vector compression
	* Vector quantization is a topline (directly optimizes the objective)
	* Binary quantization and hamming comparison 
	* Product quantization. Chunked vector quantization. Optimized vector quantization
	* Additive quantization. Extension of product quantization. Difficulty in training approximations (Residual quantization, CQ, TQ, LSQ, etc.)
	* Cost of coarse quantization vs. inverted list scanning

1. Graph based indexes
	* Early works: hierarchical k-means 
	* Neighborhood graphs. How to construct them. Nearest Neighbor Descent
	* Greedy search in Neighborhood graphs. That does not work -- need long jumps
	* HNSW. A practical hierarchical graph-based index
	* NSG. Evolving a graph k-NN graph

1. Computing Hardware and Vector Search
	* Computing platform: local vs. service / CPU vs. GPU 
	* efficient implementation of brute force search
	* distance computations for product quantization -- tradeoffs. SIMD implementation
	* Parallelization and distribution -- sharding vs. inverted list distribution
	* Using co-processors (GPUs)
	* Using a hierarchy of memory types (RAM + SSD or RAM + GPU RAM)

1. Advanced topics -- articles to be presented by students
	* Vectors in Generative AI 
	* Neural quantization 
	* Vector Databases 
	* Beyond feature extraction and indexing: neural indexing
	
1. In class project presentations

 
## Selected literature 

* Product quantization (PQ) and inverted file: [“Product quantization for nearest neighbor search”](https://hal.inria.fr/inria-00514462v2/document), Jégou & al., PAMI 2011. 

* Fast k-selection on wide SIMD architectures: [“Billion-scale similarity search with GPUs”](https://arxiv.org/abs/1702.08734), Johnson & al, ArXiv 1702.08734, 2017 

* HNSW indexing method: ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/abs/1603.09320), Malkov & al., ArXiv 1603.09320, 2016

* In-register vector comparisons: ["Quicker ADC : Unlocking the Hidden Potential of Product Quantization with SIMD"](https://arxiv.org/abs/1812.09162), André et al, PAMI'19, also used in ["Accelerating Large-Scale Inference with Anisotropic Vector Quantization"](https://arxiv.org/abs/1908.10396), Guo, Sun et al, ICML'20.

* Graph-based indexing method NSG: ["Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph"](https://arxiv.org/abs/1707.00143), Cong Fu et al, VLDB 2019.

* Local search quantization (an additive quantization method): ["Revisiting additive quantization"](https://drive.google.com/file/d/1dDuv6fQozLQFS2AJoNNFGTH499QIp_vO/view), Julieta Martinez, et al. ECCV 2016 and ["LSQ++: Lower running time and higher recall in multi-codebook quantization"](https://openaccess.thecvf.com/content_ECCV_2018/html/Julieta_Martinez_LSQ_lower_runtime_ECCV_2018_paper.html), Julieta Martinez, et al. ECCV 2018.

* Learning-based quantizer: ["Unsupervised Neural Quantization for Compressed-Domain Similarity Search"](https://openaccess.thecvf.com/content_ICCV_2019/html/Morozov_Unsupervised_Neural_Quantization_for_Compressed-Domain_Similarity_Search_ICCV_2019_paper.html), Morozov and Banenko, ICCV'19

* Neural indexing: ["Transformer Memory as a Differentiable Search Index"](https://arxiv.org/abs/2202.06991), Tan & al. ArXiV'22

* The hybrid RAM-disk index: ["Diskann: Fast accurate billion-point nearest neighbor search on a single node"](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html), Subramanya & al. NeurIPS'19

* The nearest neighbor descent method: [Efficient k-nearest neighbor graph construction for generic similarity measures](https://www.ambuehler.ethz.ch/CDstore/www2011/proceedings/p577.pdf) from Dong et al., WWW'11

## Build

On unix-like systems with the bibtex and pdflatex available you should be able to do this:


```
git clone git@github.com:edoliberty/vector-search-class-notes.git
cd vector-search-class-notes
./build
```



## Contribute

These class notes are intended to be used freely by academics anywhere, students and professors alike. Please feel free to contribute in the form of pull requests or opening issues.
