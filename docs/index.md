## Summary

Word embeddings have received a lot of attention since some Tomas Mikolov published [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) in 2013  and showed that the embeddings that the neural network learned by "reading" a large corpus of text preserved semantic relations between words. As a result, this type of embedding started being studied in more detail and applied to more serious NLP and IR tasks such as summarization, query expansion, etc... More recently, researchers and practitioners alike have come to appreciate the power of this type of approach and have started a cottage industry of modifying Mikolov's original approach to many different areas.  

In this talk we will cover the implementation and mathematical details underlying tools like word2vec and some of the applications word embeddings have found in various areas. Starting from an intuitive overview of the main concepts and algorithms underlying the neural network architecture used in word2vec we will proceed to discussing the implementation details of the word2vec reference implementation in tensorflow.
Finally, we will provide a birds eye view of the emerging field of "<anything>2vec" (phrase2vec, doc2vec, dna2vec, node2vec, etc...) methods that use variations of the word2vec neural network architecture will also be presented.

### Tentative Program

1. Neural network architecture and algorithms underlying word2vec.
* Basic intuition
* Skip-gram
* Continuous bag of words
* Hierarchical softmax
* Cross entropy
* Negative sampling
* Semantic structure and analogies
* Online sources for pre-trained embeddings

2. Brief refresher of tensorflow

3. Detailed discussion of tensorflows [reference implementation](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py)

4. Overview of word2vec variations and their applications
* [phrase2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
* [doc2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
* [dna2vec](https://arxiv.org/abs/1701.06279)
* [node2vec](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf)

### Contact

Bruno Gonçalves [[web]](http://www.bgoncalves.com) [[email]](mailto:bgoncalves@gmail.com) [[twitter]](https://twitter.com/bgoncalves)