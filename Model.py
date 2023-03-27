import googlemaps

gmaps = googlemaps.Client(key='AIzaSyAz1SL-ycn4SNhtCGFejPRvGIvykYV22t0')
radius = 1000
keyword = 'park'
#新北市-新店區
location = (24.9676, 121.53)

result = gmaps.places_nearby(location=location, radius=radius, keyword=keyword, type='餐廳', language='zh-tw')

for place in result['results']:
    if place['rating']  > 4 :
        name = place['name']
        rating = place['rating']
        address = place['vicinity']
        #opening_hours = place['opening_hours']
        #print(f'{name} - 評級: {rating} - 地址: {address} - 營業時間: {opening_hours}')
        #print(f'{name} - 評級: {rating} - 地址: {address}')