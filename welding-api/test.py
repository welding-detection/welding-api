import uuid
from django.test import TestCase
from rest_framework.test import APIClient
from .models import User


class APITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        User.objects.get_or_create(login_id='test', defaults={'password': 'test123', 'name': 'Test User'})

    def test_signup(self):
        unique_login_id = f'test_{uuid.uuid4()}'
        response = self.client.post('/signup/',
                                    {'login_id': unique_login_id, 'password': 'test123', 'name': 'Test User'})
        self.assertEqual(response.status_code, 201)

    def test_login(self):
        response = self.client.post('/login/', {'login_id': 'test', 'password': 'test123'})
        print(response.content)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.get('loginStatus'))
        self.token = response.data.get('sessionToken')

    def test_upload(self):
        # Using the token from the login test
        response = self.client.post('/upload/', {'login_id': 'test', 'file_name': 'test.jpg',
                                                 'file_url': 'http://example.com/test.jpg'},
                                    HTTP_AUTHORIZATION=f'Bearer {"justicehand1999"}')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.get('success'))

    def test_list_retrieval(self):
        response = self.client.get(f'/list/test/', HTTP_AUTHORIZATION=f'Bearer {"justicehand1999"}')
        self.assertEqual(response.status_code, 200)

    def test_detail_retrieval(self):
        inspection_id = 1  # 업로드 임시 성공
        response = self.client.get(f'/detail/{inspection_id}/', HTTP_AUTHORIZATION=f'Bearer {"justicehand1999"}')
        self.assertEqual(response.status_code, 200)
