import unittest
import os
import requests

class FlaskTests(unittest.TestCase):

	def test_a_index(self):
		responce = requests.get('http://localhost:5000')
		self.assertEqual(responce.status_code, 200)
		
	def test_b_add_user(self):
		
		params = {
			'uid': self.user['uid'],
			
			"form_type": "add_user"
		}
		responce = requests.post('http://localhost:5000', data=params)
		self.assertEqual(responce.status_code, 200)
		self.assertEqual(responce.content, 'success'.encode())
		
	
		
		
if __name__ == '__main__':
	unittest.main()		




