"""Execution deferred by user; run with unittest on the target machine."""
import io
import unittest
from PIL import Image
from app import create_app


class FakePredictor:
    def predict(self, image):
        return {'width': image.width, 'height': image.height}


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(predictor=FakePredictor(), api_key='test').test_client()

    def test_operations_and_formats(self):
        for op, expected in [('sum', 10), ('subtract', 6), ('multiply', 16), ('divide', 4)]:
            for kind in ('data', 'json'):
                response = self.client.post('/calculate', **{kind: {'Operation': op, 'Data_A': 8, 'Data_B': 2}}, headers={'X-API-Key': 'test'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json, {'Result': expected})

    def test_bad_values(self):
        for value in ('', 'NaN', 'inf', '1e999', 'abc', True):
            r = self.client.post('/calculate', json={'Operation': 'sum', 'Data_A': value, 'Data_B': 1}, headers={'X-API-Key': 'test'})
            self.assertEqual(r.status_code, 400)
        r = self.client.post('/calculate', data={'Operation': 'divide', 'Data_A': 1, 'Data_B': 0}, headers={'X-API-Key': 'test'})
        self.assertEqual(r.status_code, 400)

    def test_auth_and_upload(self):
        self.assertEqual(self.client.post('/calculate').status_code, 401)
        image = io.BytesIO()
        Image.new('RGB', (32, 40)).save(image, format='PNG')
        image.seek(0)
        r = self.client.post('/predict', data={'file': (image, 'x.png')}, headers={'X-API-Key': 'test'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json['width'], 32)
        self.assertEqual(self.client.post('/predict', data={'file': (io.BytesIO(b'bad'), 'bad.png')}, headers={'X-API-Key': 'test'}).status_code, 400)

    def test_no_model(self):
        client = create_app(api_key='').test_client()
        b = io.BytesIO()
        Image.new('RGB', (8, 8)).save(b, 'PNG')
        b.seek(0)
        self.assertEqual(client.post('/predict', data={'file': (b, 'x.png')}).status_code, 503)


if __name__ == '__main__':
    unittest.main()
