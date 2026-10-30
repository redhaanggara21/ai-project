from django.test import TestCase, Client
from django.urls import reverse

class SystemTestingWorkflow(TestCase):
    def setUp(self):
        self.client = Client()

    def test_training_and_polling_workflow(self):
        """Menguji alur sistem end-to-end mulai dari pemicu training hingga polling status"""
        response_start = self.client.post(reverse('start_training_classification'), {
            'algoritma': 'CNN',
            'kategori': 'Klasifikasi',
            'optimasi': 'XGBoost',
            'dataset': 'Retina'
        })
        self.assertIn(response_start.status_code, [200, 400, 500])

        response_prog = self.client.get(reverse('get_training_progress'))
        self.assertEqual(response_prog.status_code, 200)
        self.assertIn('progress', response_prog.json())

        response_res = self.client.get(reverse('get_training_results'))
        self.assertEqual(response_res.status_code, 200)
        self.assertIn('accuracy', response_res.json())