from predict import *
from similarity import *
from validate import *
import os
import logging
logging.disable()


def main():
    # create results folder
    os.system("mkdir results")
    os.system("mkdir results/predicted_scores_GSE")
    os.system("mkdir results/predicted_scores_GSE_other")
    os.system("mkdir results/similarity_scores_GSE")
    os.system("mkdir results/similarity_socres_GSE_all")
    os.system("mkdir results/gaps")

    # predict scores for reactions in reaction pool
    repeat = 5
    for i in range(repeat):
        get_prediction_score(name='GSE')

    # predict mean similarity between candidate reactions and existing reactions
    # get_similarity_score(name='GSE', top_N=7500)

    # predict metabolic phenotypes
    # If you only want prediction and similarity scores, comment out the following line
    # validate()


if __name__ == "__main__":
    main()
