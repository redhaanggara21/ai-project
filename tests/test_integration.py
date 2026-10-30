import os
import joblib
from django.test import TestCase, Client
from django.urls import reverse
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile

class IntegrationTestingViews(TestCase):
    def setUp(self):
        self.client = Client()
        self.model_dir = os.path.join(settings.BASE_DIR, "saved_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.dummy_filename = "cnn_xgboost_retina.pkl"
        self.dummy_path = os.path.join(self.model_dir, self.dummy_filename)
        
        class DummyModel:
            def predict(self, features):
                return [0]
                
        bundle = {
            "model": DummyModel(),
            "class_names": ["Normal", "Katarak"]
        }
        joblib.dump(bundle, self.dummy_path)

    def tearDown(self):
        if os.path.exists(self.dummy_path):
            os.remove(self.dummy_path)

    def test_run_testing_with_file(self):
        """Menguji integrasi upload file gambar, pembacaan model bundle .pkl, dan pemetaan kelas"""
        dummy_image = SimpleUploadedFile(
            name='retina_test.jpg',
            content=b'\xFF\xD8\xFF\xE0\x00\x10JFIF',
            content_type='image/jpeg'
        )

        response = self.client.post(reverse('run_testing'), {
            'select-model': self.dummy_filename,
            'test_file': [dummy_image]
        })

        self.assertEqual(response.status_code, 200)
        json_resp = response.json()
        self.assertEqual(json_resp['status'], 'success')
        self.assertEqual(json_resp['result'][0]['prediction'], 'Normal')