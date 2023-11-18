# Long Term Memory in AI - Vector Search and Databases

**NOTE:** COS 597A class times changed for Fall semester 2023. Classes will be held **9am-12noon**.

## Instructors

* [Edo Liberty](https://edoliberty.github.io), Founder and CEO of [Pinecone](https://www.pinecone.io), the world's leading Vector Database. [Publications](https://scholar.google.com/citations?user=QHS_pZAAAAAJ&hl=en).

* [Matthijs Douze](https://ai.meta.com/people/matthijs-douze/), Research Scientist at Meta. Architect and main developer of [FAISS](https://github.com/facebookresearch/faiss) the most popular and advanced open source library for vector search. [Publications](https://scholar.google.com/citations?user=0eFZtREAAAAJ&hl=en).

* Teaching assistant [Nataly Brukhim](https://www.cs.princeton.edu/~nbrukhim/) PhD sdudent working with Prof. Elad Hazan and researcher at Google AI Princeton. email: <nbrukhim@princeton.edu>. [Publications](https://scholar.google.com/citations?user=jZwEDZoAAAAJ&hl=en).

* Guest lecture by [Harsha Vardhan Simhadri](https://harsha-simhadri.org/) Senior Principal Researcher, at Microsoft Research. Creator of [DiskANN](https://github.com/Microsoft/DiskANN). [Publications](https://scholar.google.com/citations?user=bW65tuAAAAAJ&hl=en)


## Overview
Long Term Memory is a foundational capability in the modern AI Stack. At their core, these systems use vector search. Vector search is also a basic tool for systems that manipulate large collections of media like search engines, knowledge bases, content moderation tools, recommendation systems, etc. As such, the discipline lays at the intersection of Artificial Intelligence and Database Management Systems. This course will cover the theoretical foundations and practical implementation of vector search applications, algorithms, and systems. The course will be evaluated with project and in-class presentation. 

## Contribute

All class materials are intended to be used freely by academics anywhere, students and professors alike. Please contribute in the form of pull requests or by opening issues.

```
https://github.com/edoliberty/vector-search-class-notes
```

On unix-like systems (e.g. macos) with bibtex and pdflatex available you should be able to run this:

```
git clone git@github.com:edoliberty/vector-search-class-notes.git
cd vector-search-class-notes
./build
```

## Syllabus

* 9/8 - [Class 1 - Introduction to Vector Search](class_notes/Class_01_introduction.pdf) [Matthijs + Edo + Nataly]
	* Intro to the course: Topic, Schedule, Project, Grading, ...
	* Embeddings as an information bottleneck. Instead of learning end-to-end, use embeddings as an intermediate representation
	* Advantages: scalability, instant updates, and explainability
	* Typical volumes of data and scalability. Embeddings are the only way to manage / access large databases
	* The embedding contract: the embedding extractor and embedding indexer agree on the meaning of the distance. Separation of concerns.
	* The vector space model in information retrieval
	* Vector embeddings in machine learning
	* Define vector, vector search, ranking, retrieval, recall

	
* 9/15 - [Class 2 - Text embeddings](class_notes/Class_02_text_embeddings.pdf) [Matthijs]
	* 2-layer word embeddings. Word2vec and fastText, obtained via a factorization of a co-occurrence matrix. Embedding arithmetic: king + woman - man = queen, (already based on similarity search)
	* Sentence embeddings: How to train, masked LM. Properties of sentence embeddings.
	* Large Language Models: reasoning as an emerging property of a LM. What happens when the training set = the whole web


* 9/22 - [Class 3 - Image embeddings](class_notes/Class_03_image_embeddings.pdf) [Matthijs] 
	* Pixel structures of images. Early works on direct pixel indexing
	* Traditional CV models. Global descriptors (GIST). Local descriptors (SIFT and friends)Direct indexing of local descriptors for image matching, local descriptor pooling (Fisher, VLAD)
	* Convolutional Neural Nets. Off-the-shelf models. Trained specifically (contrastive learning, self-supervised learning)
	* Modern Computer Vision models 


* 9/29 - [Class 4 - Low Dimensional Vector Search](class_notes/Class_04_low_dimensional_vector_search.pdf) [Edo]
	* Vector search problem definition 
	* k-d tree, space partitioning data structures
	* Worst case proof for kd-trees
	* Probabilistic inequalities. Recap of basic inequalities: Markov, Chernoof, Hoeffding
	* Concentration Of Measure phenomena. Orthogonality of random vectors in high dimensions
	* Curse of dimensionality and the failure of space partitioning

* 10/6 - [Class 5 - Dimensionality Reduction](class_notes/Class_05_dimensionality_reduction.pdf) [Edo] 
	* Singular Value Decomposition (SVD)
	* Applications of the SVD
	* Rank-k approximation in the spectral norm
	* Rank-k approximation in the Frobenius norm
	* Linear regression in the least-squared loss
	* PCA, Optimal squared loss dimension reduction
	* Closest orthogonal matrix
	* Computing the SVD: The power method
	* Random-projection
	* Matrices with normally distributed independent entries
	* Fast Random Projections

* 10/13 - No Class - Midterm Examination Week

* 10/20 - No Class - Fall Recess

* 10/27 - [Class 6 - Approximate Nearest Neighbor Search](class_notes/Class_06_aproximate_nearest_neighbor_search.pdf) [Edo]
	* Definition of Approximate Nearest Neighbor Search (ANNS)
	* Criteria: Speed / accuracy / memory usage / updateability / index construction time 
	* Definition of Locality Sensitive Hashing and examples
	* The LSH Algorithm
	* LSH Analysis, proof of correctness, and asymptotics

* 11/3 - [Class 7 - Clustering](class_notes/Class_07_clustering.pdf) [Edo]
	* K-means clustering - mean squared error criterion.
	* Lloyd’s Algorithm
	* k-means and PCA
	* ε-net argument for fixed dimensions
	* Sampling based seeding for k-means
	* k-means++
	* The Inverted File Model (IVF)
	
* 11/10 - [Class 8 - Quantization for lossy vector compression](class_notes/Class_08_quantization.pdf) **This class will take place remotely via zoom, see the edstem message to get the link** [Matthijs]
	* Python notebook corresponding to the class: [Class_08_runbook_for_students.ipynb](class_notes/Class_08_runbook_for_students.ipynb)
	* Vector quantization is a topline (directly optimizes the objective)
	* Binary quantization and hamming comparison 
	* Product quantization. Chunked vector quantization. Optimized vector quantization
	* Additive quantization. Extension of product quantization. Difficulty in training approximations (Residual quantization, CQ, TQ, LSQ, etc.)
	* Cost of coarse quantization vs. inverted list scanning
	
* 11/17 - [Class 9 - Graph based indexes](class_notes/Class_09_graph_indexes.pdf) by Guest lecturer [Harsha Vardhan Simhadri.](https://harsha-simhadri.org/)
	* Early works: hierarchical k-means 
	* Neighborhood graphs. How to construct them. Nearest Neighbor Descent
	* Greedy search in Neighborhood graphs. That does not work -- need long jumps
	* HNSW. A practical hierarchical graph-based index
	* NSG. Evolving a graph k-NN graph	


* 11/24 - No Class - Thanksgiving Recess

* 12/1 - Class 10 - Student project and paper presentations [Edo + Nataly]


## Project 

Class work includes a final project. It will be graded based on 

1. 50% - Project submission 
1. 50% - In-class presentation 

**Projects can be in three different flavors**

* _Theory/Research_: propose a new algorithm for a problem we explored in class (or modify an existing one), explain what it achieves, give experimental evidence or a proof for its behavior. If you choose this kind of project you are expected to submit a write up.
* _Data Science/AI_: create an interesting use case for vector search using Pinecone, explain what data you used, what value your application brings, and what insights you gained. If you choose this kind of project you are expected to submit code (e.g. Jupyter Notebooks) and a writeup of your results and insights. 
* _Engineering/HPC_: adapt or add to FAISS, explain your improvements, show experimental results. If you choose this kind of project you are expected to submit a branch of FAISS for review along with a short writeup of your suggested improvement and experiments. 


**Project schedule**  

* 11/24 - One-page project proposal approved by the instructors
* 12/1 - Final project submission
* 12/1 - In-class presentation


**Some more details**

* Project Instructor: Nataly <nbrukhim@princeton.edu>
* Projects can be worked on individually, in teams of two or at most three students.
* Expect to spend a few hours over the semester on the project proposal. Try to get it approved well ahead of the deadline. 
* Expect to spent 3-5 _full days_ on the project itself (on par with preparing for a final exam) 
* In class project project presentation are 5 minutes _per student_ (teams of two students present for 10 minutes. Teams of three, 15 minutes).
 
## Selected Literature 

* [A fast random sampling algorithm for sparsifying matrices](http://dx.doi.org/10.1007/11830924_26) - Arora, Sanjeev and Hazan, Elad and Kale, Satyen - 2006
* [A Randomized Algorithm for Principal Component Analysis](http://dx.doi.org/10.1137/080736417) - Vladimir Rokhlin and Arthur Szlam and Mark Tygert - 2009
* A search structure based on kd trees for efficient ray tracing - Subramanian, KR and Fussel, DS - 1990
* A Short Proof for Gap Independence of Simultaneous Iteration - Edo Liberty - 2016
* Accelerating Large-Scale Inference with Anisotropic Vector Quantization - Ruiqi Guo and Philip Sun and Erik Lindgren and Quan Geng and David Simcha and Felix Chern and Sanjiv Kumar - 2020
* [Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-28-2015) - 2015
* [An Algorithm for Online K-Means Clustering](https://epubs.siam.org/doi/abs/10.1137/1.9781611974317.7) - Edo Liberty and Ram Sriharsha and Maxim Sviridenko
* An Almost Optimal Unrestricted Fast Johnson-Lindenstrauss Transform - Nir Ailon and Edo Liberty - 2011
* An elementary proof of the Johnson-Lindenstrauss lemma - S. DasGupta and A. Gupta - 1999
* Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform - Nir Ailon and Bernard Chazelle - 2006
* Billion-scale similarity search with GPUs - Jeff Johnson and Matthijs Douze and Herv{\'e} J{\'e}gou - 2017
* Clustering Data Streams: Theory and Practice - Sudipto Guha and Adam Meyerson and Nina Mishra and Rajeev Motwani and Liadan O'Callaghan - 2003
* [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) - Jayaram Subramanya, Suhas and Devvrit, Fnu and Simhadri, Harsha Vardhan and Krishnawamy, Ravishankar and Kadekodi, Rohan - 2019
* Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs - Yu. A. Malkov and D. A. Yashunin - 2018
* [Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures](https://doi.org/10.1145/1963405.1963487) - Dong, Wei and Moses, Charikar and Li, Kai - 2011
* Even Simpler Deterministic Matrix Sketching - Edo Liberty - 2022
* Extensions of Lipschitz mappings into a Hilbert space - W. B. Johnson and J. Lindenstrauss - 1984
* Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph - Cong Fu and Chao Xiang and Changxu Wang and Deng Cai - 2018
* [Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions](http://dx.doi.org/10.1137/090771806) - Halko, N. and Martinsson, P. G. and Tropp, J. A. - 2011
* Invertibility of random matrices: norm of the inverse - Mark Rudelson - 2008
* K-means clustering via principal component analysis - Chris H. Q. Ding and Xiaofeng He - 2004
* k-means++: the advantages of careful seeding - David Arthur and Sergei Vassilvitskii - 2007
* Least squares quantization in pcm - Stuart P. Lloyd - 1982
* [LSQ++: Lower Running Time and Higher Recall in Multi-Codebook Quantization](https://doi.org/10.1007/978-3-030-01270-0_30) - Martinez, Julieta and Zakhmi, Shobhit and Hoos, Holger H. and Little, James J. - 2018
* [Multidimensional binary search trees used for associative searching](http://doi.acm.org/10.1145/361002.361007) - Bentley, Jon Louis - 1975
* [Near-Optimal Entrywise Sampling for Data Matrices](https://proceedings.neurips.cc/paper_files/paper/2013/file/6e0721b2c6977135b916ef286bcb49ec-Paper.pdf) - Achlioptas, Dimitris and Karnin, Zohar S and Liberty, Edo - 2013
* Pass Efficient Algorithms for Approximating Large Matrices - Petros Drineas and Ravi Kannan - 2003
* Product Quantization for Nearest Neighbor Search - Jegou, Herve and Douze, Matthijs and Schmid, Cordelia - 2011
* QuickCSG: Arbitrary and faster boolean combinations of n solids - Douze, Matthijs and Franco, Jean-S{\'e}bastien and Raffin, Bruno - 2015
* [Quicker {ADC} : Unlocking the Hidden Potential of Product Quantization With {SIMD](https://doi.org/10.1109%2Ftpami.2019.2952606) - Fabien Andre and Anne-Marie Kermarrec and Nicolas Le Scouarnec - 2021
* [Random Projection Trees and Low Dimensional Manifolds](https://doi.org/10.1145/1374376.1374452) - Dasgupta, Sanjoy and Freund, Yoav - 2008
* [Randomized Algorithms for Low-Rank Matrix Factorizations: Sharp Performance Bounds](http://dx.doi.org/10.1007/s00453-014-9891-7) - Witten, Rafi and Cand\`{e}s, Emmanuel - 2015
* [Randomized Block Krylov Methods for Stronger and Faster Approximate Singular Value Decomposition](http://papers.nips.cc/paper/5735-randomized-block-krylov-methods-for-stronger-and-faster-approximate-singular-value-decomposition) - Cameron Musco and Christopher Musco - 2015
* [Revisiting Additive Quantization](https://api.semanticscholar.org/CorpusID:7340738) - Julieta Martinez and Joris Clement and Holger H. Hoos and J. Little - 2016
* [Sampling from large matrices: An approach through geometric functional analysis](http://doi.acm.org/10.1145/1255443.1255449) - Rudelson, Mark and Vershynin, Roman - 2007
* Similarity estimation techniques from rounding algorithms - Moses Charikar - 2002
* Similarity Search in High Dimensions via Hashing - Aristides Gionis and Piotr Indyk and Rajeev Motwani - 1999
* Simple and Deterministic Matrix Sketching - Edo Liberty - 2012
* Smaller Coresets for k-Median and k-Means Clustering - S. {Har-Peled} and A. Kushal - 2005
* Sparser Johnson-Lindenstrauss transforms - Daniel M. Kane and Jelani Nelson - 2012
* Sparsity Lower Bounds for Dimensionality Reducing Maps - Jelani Nelson and Huy L. Nguyen - 2012
* Spectral Relaxation for K-means Clustering - Hongyuan Zha and Xiaofeng He and Chris H. Q. Ding and Ming Gu and Horst D. Simon - 2001
* Streaming k-means approximation - Nir Ailon and Ragesh Jaiswal and Claire Monteleoni - 2009
* Strong converse for identification via quantum channels - Rudolf Ahlswede and Andreas Winter - 2002
* Transformer Memory as a Differentiable Search Index - Yi Tay and Vinh Q. Tran and Mostafa Dehghani and Jianmo Ni and Dara Bahri and Harsh Mehta and Zhen Qin and Kai Hui and Zhe Zhao and Jai Gupta and Tal Schuster and William W. Cohen and Donald Metzler - 2022
* [Unsupervised Neural Quantization for Compressed-Domain Similarity Search](https://doi.ieeecomputersociety.org/10.1109/ICCV.2019.00313) - S. Morozov and A. Babenko - 2019
* [Worst-Case Analysis for Region and Partial Region Searches in Multidimensional Binary Search Trees and Balanced Quad Trees](https://doi.org/10.1007/BF00263763) - Lee, D. T. and Wong, C. K. - 1977
