import unittest
import os
import json
from youyakuman_train_and_deploy import model_fn, inference
from lib.ModelExecutor import ModelExecutor
from lib.Preprocessor import Preprocessor

#from sagemaker.content_types import CONTENT_TYPE_JSON, CONTENT_TYPE_NPY


class TestYouyakuman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")
        
        # get model
        base_path = "/opt/ml/model/code"
        model_dir = os.path.join(base_path, "models", "Japanese")
        model_infos = model_fn(model_dir)
        cls.model_infos = model_infos
    
    def setUp(self):
        print("setup")
    
    def tearDown(self):
        print("tearDown")
    
    def test_inference(self):
        # start to inference
        data = "これはテストです。"
        result = inference(net=self.model_infos, data=data, num_of_summaries=1)
        
        # check result
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        self.assertEqual(1, len(result[0]))
        self.assertEqual("これはテストです", result[0][0])


if __name__ == "__main__":
    unittest.main()
