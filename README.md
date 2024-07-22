# Random Indexing

## Note

This project was developed on they university's GitLab, large files could not be uploaded to GitHub, and as they were in the history of commits, this repository does not have it.


## Description

This project deals with random indexing as one of the approaches to represent words/documents through vectors.
It creates random index vectors for each unique word of the document 
to later use them as a part of another word's context vector if the words are located within a specific window length.
The project encompasses intrinsic and extrinsic evaluation of vectors.
In intrinsic evaluation annotated data is compared to cosine similarity scores between context vectors of two words,
for that spearman coefficient is used. 
In extrinsic evaluation document vectors are calculated based on context vectors of tokens of the document,
which are then used to predict genre of a specific document. 

## Structure

`preprocessing` contains files to preprocess corpus.

`random_indexing_evaluation`contains files to build a model and evaluate it.

`model` contains subfolders with models.



## Annotation data

To be able to run `evaluate_ratings.py` first annotation data sets must be loaded. 

Wordsim: http://alfonseca.org/eng/research/wordsim353.html

Simlex: https://fh295.github.io/simlex.html

Global path to an annotation file will be used as a parameter for `evaluate_ratings.py`.


## Requirements
`python3.7`

`nltk` (see https://www.nltk.org/)

`scikitlearn` (see https://pypi.org/project/scikit-learn/)

`numpy` (see https://pypi.org/project/numpy/)




## Usage

Call all the files from root folder (so that random_indexing is a subfolder then)

Preprocessed coprus is already saved as `clean_documents.pkl`
 in  `random_indexing`. But you can also run `preprocess_documents.py` if you wish.
 
    `python -m random_indexing.preprocessing.preprocess_documents --file clean_documents.pkl`
    
 The file will save preprocessed data in `clean_documents.pkl`
    
To run `split_data.py`

    `python -m random_indexing.random_indexing_evaluation.split_data --file clean_documents.pkl`
   
The file will create `train_data.pkl`, `dev_data.pkl`, `test_data.pkl`.

To run `build_model.py`


    `python -m random_indexing.random_indexing_evaluation.build_model --file clean_documents.pkl --size 100 --l 1 --r 2 --dir model_new`

It will save new model in models/model_new.

To run  `evaluate_ratings.py` for wordsim

    `python -m random_indexing.random_indexing_evaluation.evaluate_ratings --model "models/model_new/model_new.pkl" --ratings  "/Users/olhasvezhentseva/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt"`

 Change  "/Users/olhasvezhentseva/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt" 
 to your global path of annotated file. 
 
 To run  `evaluate_ratings.py` for simlex
 
    `python -m random_indexing.random_indexing_evaluation.evaluate_ratings --model "models/model_new/model_new.pkl" --ratings "/Users/olhasvezhentseva/SimLex-999/SimLex-999.txt" --source 'simlex'`
 
 To run `build_document_vectors.py`
 
     `python -m random_indexing.random_indexing_evaluation.build_document_vectors --data "clean_documents.pkl" --model "models/model_new/model_new.pkl"`
 It will save 6 files such as vectors_train, genres_train, vectors_dev, genres_dev, vectors_test, genres_test in  random_indexing.
 
 To run `evaluate_document_vectors.py`
 
     `python -m random_indexing.random_indexing_evaluation.evaluate_document_vectors -t vectors_train -l genres_train -d vectors_dev -g genres_devl`
 

## Contact
Olha Svezhentseva olha.svezhentseva@uni-potsdam.de