from preprocessing.load_dataset import load_webqsp, load_grailqa
from preprocessing.score_path import run_score_path
from preprocessing.search_to_get_path import run_search_to_get_path
from preprocessing.negative_sampling import run_negative_sampling

def run():
    load_grailqa()
    print("Load done!")
    run_search_to_get_path()
    print("Run search to get path done!")
    run_score_path()
    print("Run score path done!")
    run_negative_sampling()
    print("Run negative sampling done!")

if __name__ == '__main__':
    run()
