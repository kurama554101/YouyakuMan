import glob
import os
import pickle
import codecs
import shutil


class AbstractDatasetCreator:
    # TODO : imp
    pass


class LivedoorDatasetCreator:
    def __init__(self):
        self.file_list = []
        self.exclusive_txt_name = ["README.txt", "CHANGES.txt", "LICENSE.txt"]
    
    def exists(self, dataset_dir:str):
        return os.path.exists(dataset_dir)
    
    def remove(self, dataset_dir:str):
        shutil.rmtree(dataset_dir)
    
    def load(self, origin_dir:str):
        tmp_list = glob.glob(os.path.join(origin_dir, "**", "*.txt"), recursive=True)
        for filepath in tmp_list:
            if os.path.basename(filepath) in self.exclusive_txt_name:
                # this filepath is not added because it is excluded file.
                continue
            self.file_list.append(filepath)
    
    def create(self, dataset_dir:str):
        for filepath in self.file_list:
            with codecs.open(filename=filepath, mode="r", encoding="utf-8") as f:
                text_list = f.readlines()
                summary_list = []
                body = ""
                for index in range(len(text_list)):
                    # 0-line : article url
                    # 1-line : article date
                    # above data is ignored
                    if index == 0 or index == 1:
                        continue
                    
                    # 2-line : article title -> article summary
                    if index == 2:
                        summary_list.append(text_list[index])
                        continue
                    
                    # 3-lines down : body data
                    if index >= 3:
                        body = body + text_list[index]
                
                # save data into pickle file
                data_dict = self.__convert_dict(body, summary_list)
                file_name = os.path.splitext(os.path.basename(filepath))[0] + ".pickle"
                file_path = os.path.join(dataset_dir, file_name)
                with open(file_path, "wb") as pf:
                    pickle.dump(data_dict, pf)
                     
    
    def __convert_dict(self, body:str, summary_list:list):
        data_dict = {}
        data_dict["body"]    = body
        data_dict["summary"] = ""
        
        for index in range(len(summary_list)):
            data_dict["summary"] = data_dict["summary"] + summary_list[index]
            if index != len(summary_list) - 1:
                data_dict["summary"] = data_dict["summary"] + "<sep>"
        
        return data_dict
