from django.test import TestCase, Client
from django.urls import reverse

class UnitTestingViews(TestCase):
    def setUp(self):
        self.client = Client()

    def test_start_training_invalid_method(self):
        """Memastikan endpoint start training menolak method GET (harus 405)"""
        response = self.client.get(reverse('start_training_classification'))
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.json()['status'], 'error')

    def test_run_testing_no_model(self):
        """Memastikan sistem merespons error jika model belum dipilih saat testing"""
        response = self.client.post(reverse('run_testing'), {})
        self.assertEqual(response.json()['status'], 'error')
        self.assertIn('Belum ada model', response.json()['msg'])

    def test_kfold_invalid_method(self):
        """Memastikan endpoint K-Fold menolak method GET"""
        response = self.client.get(reverse('run_kfold_evaluation'))
        self.assertEqual(response.status_code, 405)