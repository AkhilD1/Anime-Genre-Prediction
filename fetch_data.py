import json
import requests
import time


endpoint = 'https://api.myanimelist.net/v2/anime/{}?fields=id,title,synopsis,genres'
# Replace your bearer token below
bearer_token = '<token-string>'
headers = {'Authorization': 'Bearer {}'.format(bearer_token)}

# Fetch the first 15000 ids
for i in range(15000):
    tmp = requests.get(endpoint.format(i), headers = headers)
    # Since not all numbers are valid ids, the request can return an error
    if tmp.status_code == 200:
        # Write the data to a file
        data = tmp.json()
        # Some responses have an extra entry for 'main_picture'
        # Discard it for consistent results
        if data.get('main_picture'):
            del(data['main_picture'])
        # Discard any entries that have synopsis length less than 200
        if len(data['synopsis']) > 200:
            # Dump the json data into a file for subsequent usage
            with open('{}'.format(i), 'w') as f:
                f.write(json.dumps(data))
    # Sleep to prevent too many requests to the api
    time.sleep(1)
