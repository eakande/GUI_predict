
import requests

url = 'http://localhost:3000/results'
r = requests.post(url,json={'Food':0, 'PMI(t-1)':0, 'ATM':0, 'WEBPAY':0, 'Insecurity':0,})

print(r.json())

