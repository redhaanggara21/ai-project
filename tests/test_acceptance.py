from django.test import TestCase, Client
from django.urls import reverse

class AcceptanceTestingContract(TestCase):
    def setUp(self):
        self.client = Client()

    def test_kfold_frontend_contract(self):
        """Memastikan struktur data JSON K-Fold sesuai dengan ekspektasi komponen UI dashboard"""
        response = self.client.post(reverse('run_kfold_evaluation'), {
            'k_fold': 5
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get('status'), 'success')
        self.assertIn('k', data)
        self.assertIn('scores', data)
        self.assertIn('average_accuracy', data)
        self.assertIn('average_loss', data)
        self.assertIsInstance(data['scores'], list)
        self.assertEqual(len(data['scores']), 5)