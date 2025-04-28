class Config:
    
    def __init__(self) -> None:
        self.preprocessing = {
            "step0": {
                # "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/data/WebQSP/data/WebQSP.train.json",

                # TRAIN SET
                "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/data/GrailQA/data/grailqa_v1.0_train.json",
                "dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step0.json",

                # DEV SET
                # "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/data/GrailQA/data/grailqa_v1.0_dev.json",
                # "dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step0_dev.json",

                "dump_data_folder": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing",
            },
            "step1": {
                "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step0.json",
                "dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step1.json"
            },
            "step2": {
                "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step1.json",
                "dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step2.json"
            },
            "step3": {
                "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step2.json",
                "dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/train.csv",

                # "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/step2.json",
                # "dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/train.csv",

                # "sup_load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/supervised_data_train.json",
                # "sup_dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/sup_train.csv",

                "unsup_load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/unsupervised_data.json",
                "unsup_dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/unsup_train.csv",

                "dump_data_folder": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever",
            }
        }
        self.train_retriever = {
            "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/train.csv",
            "dump_model_path": "/data/user_data/dhruvgu2/model_ckpt/SimBERT",

            # "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/train.csv",
            # "dump_model_path": "/data/user_data/dhruvgu2/model_ckpt/SimBERT",

            # "sup_load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/sup_train.csv",
            # "sup_dump_model_path": "/data/user_data/dhruvgu2/model_ckpt/sup_SimBERT",
            
            # "unsup_load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/unsup_train.csv",
            # "unsup_dump_model_path": "/data/user_data/dhruvgu2/model_ckpt/unsup_SimBERT"
        }
        self.retriever_model_ckpt = self.train_retriever["dump_model_path"]
        self.retrieve_subgraph = {
            # "load_data_folder": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/data/origin_nsm_data/webqsp", # ORIGINAL
            "load_data_folder": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/preprocessing/",
            "dump_data_folder": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/reader_data/grailqa"
        }
        self.train_reader = {
            "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/reader_data/grailqa/",
            "dump_model_path": "/data/user_data/dhruvgu2/model_ckpt/nsm/",
        }
        self.retriever_finetune = {
            "checkpoint_dir": "/data/user_data/dhruvgu2/model_ckpt/nsm/",
            "load_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/reader_data/grailqa/",
            "dump_model_path": "/data/user_data/dhruvgu2/model_ckpt/SimBERT/",
            "dump_data_path": "/home/dhruvgu2/SubgraphRetrievalKBQA/results/retriever/",
        }

cfg = Config()
