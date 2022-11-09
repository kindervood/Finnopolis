import requests


data = {'address1': "images/snoop1.jpg", "address2":"images/durov1.jpg"}

#Works for /analyze, /find, /getsimilar also
a = str(requests.get('http://127.0.0.1:5000/check', json=data).content)[2:-2]

print(a)
