import os, sys, time
import csv
from selenium import webdriver
# from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from operator import itemgetter

# os.environ['MOZ_HEADLESS'] = '1'
# binary = FirefoxBinary('/usr/bin/firefox', log_file=sys.stdout)

# two = sys.argv[1]
def clean_data(data):
	try:
		return data[0].text
	except IndexError:
		return 0


def parser_oyo(driver):
	# driver = webdriver.Firefox(firefox_binary=binary)
	# driver.get("https://www.oyorooms.com/oyos-in-kathmandu")

	time.sleep(5)

	hotels_data = []

	hotels_list = driver.find_elements_by_class_name("newHotelCard")
	for hotels in hotels_list:
		hotel_name = hotels.find_elements_by_class_name("newHotelCard__hotelName")
		hotel_location =  hotels.find_elements_by_class_name("newHotelCard__hotelAddress")

		hotel_price_detail = hotels.find_elements_by_class_name("newHotelCard__pricing")
		price = hotel_price_detail[0].text
		hotel_price =  int(price.split(" ")[1])


		hotel_not_discounted_amount = hotels.find_elements_by_class_name("newHotelCard__revisedPricing")
		hotel_discount_percentage = hotels.find_elements_by_class_name("newHotelCard__discount")
		hotel_rating = hotels.find_elements_by_class_name("hotelRating__value")
		hotel_rating_remarks = hotels.find_elements_by_class_name("hotelRating__subtext")

		original_price = clean_data(hotel_not_discounted_amount)
		disc_perc = clean_data(hotel_discount_percentage)
		rating = clean_data(hotel_rating)
		remarks = clean_data(hotel_rating_remarks)

		data = {
			"Name": hotel_name[0].text,
			"Location": hotel_location[0].text,
			"Price after Disc": hotel_price,
			"Original Price": original_price,
			"Disc Percentage": disc_perc,
			"Rating": rating,
			"Remarks": remarks
			}
		print(data)
		hotels_data.append(data)
	# del os.environ['MOZ_HEADLESS'] 
	return hotels_data

def write_data_to_csv(parsed_data, csv_columns, csv_file):
	try:
		with open(csv_file, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
			writer.writeheader()
			for data in parsed_data:
				writer.writerow(data)
	except IOError:
		print("I/O error") 



if __name__ == '__main__':
	url = sys.argv[1]
	driver = webdriver.Chrome()
	driver.get(url)
	parsed_data = parser_oyo(driver)

	#get next pages data
	try:
		nextpageButton = driver.find_elements_by_class_name("btn-next")[0]
		while(nextpageButton != []):
			next_page = nextpageButton.click()
			next_page_data = parser_oyo(driver)
			parsed_data += next_page_data
			nextpageButton = driver.find_elements_by_class_name("btn-next")[0]
	except IndexError:
		pass

	driver.close()

	csv_columns = ['Name','Location','Price after Disc', 'Original Price', 'Disc Percentage', 'Rating', 'Remarks']
	csv_file = "Hotels List.csv"
	write_data_to_csv(parsed_data, csv_columns, csv_file) 

	data_sorted_by_price = sorted(parsed_data, key=itemgetter('Price after Disc'))
	sorted_csv_file = "Hotel List sorted by price.csv"
	write_data_to_csv(data_sorted_by_price, csv_columns, sorted_csv_file)
  
  
  outout:
   # Hotel lists
    Name	Location	Price after Disc	Original Price	Disc Percentage	Rating	Remarks
OYO 265 Hotel Black Stone	Bhattedanda Devistan Rd, Kavre Nitya Chandeshwor, Kathmandu	833	NPR 2178	62% OFF	0	0
OYO 287 Hotel Buddha Palace	Mahadev Marg, Kathmandu	1000	NPR 2000	50% OFF	4.3	Very Good
OYO 237 Hotel S Galaxy	Kalanki, Kathmandu	667	NPR 1612	59% OFF	4.1	Very Good
OYO 251 Siddhi Binayak Guest House	Milan Tole, New Buspark, Kathmandu	667	NPR 1334	50% OFF	4.8	Excellent
OYO 202 Hotel Kanchenjunga	Purano Bagar Marg, Near Napi Karyalaya, Kathmandu	733	NPR 1467	50% OFF	4.5	Excellent
OYO 222 Hotel New Himalayan	Mitranagar, New Buspark, Kathmandu	800	NPR 2077	61% OFF	3.9	Good
OYO 260 Hotel Cultural Inn	Sinamangal, Manpowerbazar, Kathmandu	800	NPR 2970	73% OFF	4.6	Excellent
OYO 160 Hotel Shraddha Palace	Gaushala chwok near by Pashupatinath Temple, Kathmandu	867	NPR 2578	66% OFF	4.5	Excellent
OYO 247 Hotel North Pole Inn	Araniko Highway, Kathmandu	867	NPR 2701	68% OFF	4.5	Excellent
OYO 210 Andes Hotel Pvt Ltd	Ward No16, Paknajol Marg, Kathmandu	900	NPR 2604	65% OFF	3.8	Good
OYO 252 Hotel Black Stone	Birendra Aishwarya Marg, Kathmandu	1000	NPR 3459	71% OFF	4.2	Very Good
OYO 279 City Gaon Resort	Bachamari, Nankhel-9, Araniko Highway, Kathmandu	1467	NPR 3517	58% OFF	0	0
OYO 243 Saypatri Guest House	Mitranagar-26 Naya Buspark, Kathmandu	667	NPR 1570	58% OFF	4.5	Excellent
OYO 149 Kalpa Brikshya Hotel	Pingalasthan, Jamuna Galli, Kathmandu	733	NPR 1760	58% OFF	4.1	Very Good
OYO 161 Ram Janaki Hotel	Sambhu Marg, Airport-9, Kathmandu	800	NPR 2039	61% OFF	4.7	Excellent
OYO 233 Waling Fulbari Guest House	Mitranagar-26, Gondhuli Marg, New Buspark, Kathmandu	800	NPR 3169	75% OFF	4.6	Excellent
OYO 148 hotel green orchid	paknajol road , thamel , Kathmandu	833	NPR 2277	63% OFF	4.1	Very Good
OYO 232 Hotel Titanic Pvt Ltd	Mhepi Marg-29, Kathmandu	867	NPR 2695	68% OFF	3.9	Good
OYO 135 Lost Garden Apartment and Guest House	Lazimpat, Kathmandu, Kathmandu	900	NPR 1800	50% OFF	4.2	Very Good
OYO 281 Twins Kitchen	B.P.Highway, Hurkha, Dhulikhel, Kathmandu	917	NPR 2933	69% OFF	3.9	Good
OYO 166 Kathmandu Tourist Home	Saat Ghumti Marg, Kathmandu	933	NPR 2604	64% OFF	4.1	Very Good
OYO 200 Tibet Peace Guest House	Ward No16, Paknajol Marg, Kathmandu	933	NPR 1867	50% OFF	4.1	Very Good
OYO 236 Hotel Beli Nepal	Dhalko Marga, Kathmandu	933	NPR 2472	62% OFF	3.9	Good
OYO 276 White Orchid Resort	Araniko Highway, Bansghari, Bhaktapur.(On the Way to Ktm Fun Valley), Kathmandu	933	NPR 2510	63% OFF	3.9	Good
OYO 105 Hotel Travel INN	GAUSHALA, Kathmandu	960	NPR 4352	78% OFF	4.2	Very Good
OYO 11457 Kathmandu Resort Hotel	Sagarmatha Bazar, Kathmandu	1000	NPR 2000	50% OFF	4.2	Very Good
OYO Home 167 Adventure	Bhagwan Bahal, Amrit Marg Thamel - 26, Kathmandu	1000	NPR 2513	60% OFF	4.4	Very Good
OYO 155 Sankata Hotel & Apartment	Chhusya Galli, thamel, Kathmandu	1067	NPR 2134	50% OFF	4.2	Very Good
OYO 275 Sunshine Garden Resort	Bhatey Dhikor-9 Bhakatapur, Kathmandu	1067	NPR 2134	50% OFF	3.7	Good
OYO 238 Mustang Thakali Kitchen And Guest House	Boudha 6 - Jorpati Rd, Kathmandu	1133	NPR 2670	58% OFF	4	Very Good
OYO 195 Global Hotel And Restaurant	Gaurighat, Kathmandu	933	NPR 1867	50% OFF	0	0
OYO 231 Hotel Magnificent View	Keshar Mahal Marg, Kathmandu	933	NPR 2675	65% OFF	3.6	Good
OYO 266 Hotel Grand Stupa	Tushal Chowk, Boudha, Kathmandu	933	NPR 3370	72% OFF	4.3	Very Good
OYO 123 Hotel Prince Plaza	Kathmandu, Nepal, Kathmandu 44600, Nepal, Kathmandu	960	NPR 1920	50% OFF	4.5	Excellent
OYO 158 Hotel Premium	Paknajol Rd, Kathmandu, Kathmandu	967	NPR 1934	50% OFF	4.1	Very Good
OYO 137 Hotel Pranisha Inn	Shambhu marg, Road no. 2 Airport Road, Kathmandu	1000	NPR 2000	50% OFF	4.1	Very Good
OYO 256 Mount Princess Hotel	Araniko Highway, Kathmandu	1000	NPR 2644	62% OFF	3.5	Good
OYO 174 Shiva Shankar Hotel	Pashupati Nath Road, Kathmandu, Kathmandu	1067	NPR 3520	70% OFF	4.2	Very Good
OYO 120 Hotel Tayoma	J P Road, Thamel, Kathmandu	1120	NPR 2240	50% OFF	3.8	Good
OYO 255 White Zambala Hotel	Dharatol, Boudha-6, Kathmandu	1133	NPR 2670	58% OFF	4.2	Very Good
OYO 11468 Kathmandu Embassy Hotel	Surya Mukhi Marga, Kathmandu 44600, Nepal, Kathmandu	1200	NPR 3517	66% OFF	4	Very Good
OYO 153 Aster Hotel Nepal	Bharma Kumari Margh, Jyatha, Thamel, Kathmandu	1200	NPR 3666	67% OFF	4.5	Excellent
OYO 219 Hotel Royal Kusum	pingalasthan, 8, à¤à¤¾à¤ à¤®à¤¾à¤¡à¥à¤, Kathmandu	1200	NPR 2400	50% OFF	3.9	Good
OYO 156 Hotel Sweet Town	Thamel Bhagawati Marg, Kathmandu	1440	NPR 3534	59% OFF	4.3	Very Good
OYO 11474 Gangaur Regency Boutique Hotel	Durbarmarg, Kathmandu	1533	NPR 3067	50% OFF	4.4	Very Good
OYO 172 Hotel Deepshree	Airport , Sinamangal , Kathmandu	1734	NPR 3334	48% OFF	4.5	Excellent
OYO 297 Hotel Aayam	Ganbihar Galli, Bagdurbar, Sundhara, Kathmandu	856	NPR 1713	50% OFF	0	0
OYO 273 Hotel Rara Palace	Mitranagar-26 , Naya Buspark, Kathmandu	880	NPR 1761	50% OFF	0	0
OYO 295 Asha Lodges	Araniko Highway, Dhulikhel, Kathmandu	1000	NPR 2000	50% OFF	0	0
OYO 267 Hotel Tanahun Vyas	à¤à¥à¤à¤à¤¬à¥à¤ à¤à¤à¥à¤°à¤ªà¤¥ à¤à¥à¤°à¤¾à¤ à¤à¤¤à¥à¤¤à¤°, Kathmandu	867	NPR 2191	60% OFF	3.8	Good
OYO 136 Hotel Om Plaza	Airport, Tilganga, Kathmandu	1200	NPR 3793	68% OFF	4.2	Very Good
OYO 208 Mount Gurkha Palace	sundhara, Bagdurbar, peaceline #44, Kathmandu	1200	NPR 2827	58% OFF	3.4	Fair
OYO 246 Tensar Hotel	Tusal Marg 6, Boudha, Kathmandu	1233	NPR 2467	50% OFF	4.4	Very Good
OYO 131 Hotel Milarepa	Thamel, Kathmandu, Kathmandu	1467	NPR 3707	60% OFF	4.4	Very Good
OYO 234 Nepal Tara	Narsingh Chowk Marg, Kathmandu, Kathmandu	1600	NPR 3927	59% OFF	4.3	Very Good
OYO 278 Hotel View Bhaktapur	Araniko Highway, Suryabinayak, Kathmandu	4400	NPR 7333	40% OFF	0	0
OYO 264 Hote Antique Kutty	Amrit Marg, Kathmandu	867	NPR 1734	50% OFF	0	0
OYO 293 Royal Bouddha Hotel	Mahankal, Boudhanath-6, Kathmandu	933	NPR 1867	50% OFF	0	0
OYO 280 Hob Nob Garden Resort	Jagati Bridge, Araniko Highway, Bhaktapur, Kathmandu	1067	NPR 2134	50% OFF	0	0
OYO 129 Hotel Himalaya Darshan	Kesar Mahal Road, Thamel, Kathmandu	733	NPR 2534	71% OFF	3.7	Good
OYO 142 CO International Guest House	Gaushala, Kathmandu, opposite to Pashupatinath temple entry gate, Kathmandu	733	NPR 1980	63% OFF	3.7	Good
OYO 171 Hotel Fewa Darshan	Sora Khutte, Kathmandu	800	NPR 2011	60% OFF	3.2	Fair
OYO 144 Hotel Zhonghau	Kathmandu, Satghumti, Kathmandu	867	NPR 1734	50% OFF	4.1	Very Good
OYO 135 Lost Garden Apartment and Guest House	Lazimpat, Kathmandu, Kathmandu	900	NPR 1800	50% OFF	4.2	Very Good
OYO 104 Hotel Baltic INN	Tribhuvan International Airport, Sinamangal, Kathmandu	1000	NPR 2714	63% OFF	4	Very Good
OYO 11459 Osho Holiday INN	Amrit Marg, Kathmandu 44600, Nepal, Kathmandu	1133	NPR 2267	50% OFF	3.5	Good
OYO 207 Hotel Cirrus	Karvepalanchok, Kathmandu	1200	NPR 2971	60% OFF	3.7	Good
OYO 152 Kathmandu Airport Hotel	Ring Road Near Kathmandu Airport, Kathmandu	3360	NPR 4960	32% OFF	4.8	Excellent
OYO 139 Hotel Sujata Inn	shambhu marga , airport road., Kathmandu	65067	0	0	0	0
OYO 145 Sirahali Khusbu Hotel &Lodge	Tilganga, Kathmandu	733	NPR 1467	50% OFF	4.1	Very Good
OYO 146 Somewhere Hotel & Resturant	Opposite to Airport Gate, Beside Global IME Bank, Kalimatidol, Kathmandu	833	NPR 1667	50% OFF	3.9	Good
OYO 175 Hotel Felicity	Amrit Marg, Kathmandu	867	NPR 2041	58% OFF	3.9	Good
OYO 169 Hotel Cosmic	Z Street, Thamel, Kathmandu	933	NPR 3080	70% OFF	3.8	Good
OYO 209 Tibet Peace Inn	Paknajol Marg, Kathmandu	1000	NPR 2000	50% OFF	4.5	Excellent
OYO 206 Mount View Homes	Boudhha, Kathmandu	1167	NPR 2334	50% OFF	4.7	Excellent
OYO 159 Hotel Highlander	Thamel, Kathmandu	1600	NPR 3927	59% OFF	3.8	Good
OYO 217 Shiva Tirupati Hotel	Ward No. 29, Putali Sadak, Kathmandu, Kathmandu	6400	NPR 9427	32% OFF	0	0
OYO 199 Hotel Access Nepal	Thamel, Kathmandu	81734	0	0	4.8	Excellent


#hotel list by price

Name	Location	Price after Disc	Original Price	Disc Percentage	Rating	Remarks
OYO 237 Hotel S Galaxy	Kalanki, Kathmandu	667	NPR 1612	59% OFF	4.1	Very Good
OYO 251 Siddhi Binayak Guest House	Milan Tole, New Buspark, Kathmandu	667	NPR 1334	50% OFF	4.8	Excellent
OYO 243 Saypatri Guest House	Mitranagar-26 Naya Buspark, Kathmandu	667	NPR 1570	58% OFF	4.5	Excellent
OYO 202 Hotel Kanchenjunga	Purano Bagar Marg, Near Napi Karyalaya, Kathmandu	733	NPR 1467	50% OFF	4.5	Excellent
OYO 149 Kalpa Brikshya Hotel	Pingalasthan, Jamuna Galli, Kathmandu	733	NPR 1760	58% OFF	4.1	Very Good
OYO 129 Hotel Himalaya Darshan	Kesar Mahal Road, Thamel, Kathmandu	733	NPR 2534	71% OFF	3.7	Good
OYO 142 CO International Guest House	Gaushala, Kathmandu, opposite to Pashupatinath temple entry gate, Kathmandu	733	NPR 1980	63% OFF	3.7	Good
OYO 145 Sirahali Khusbu Hotel &Lodge	Tilganga, Kathmandu	733	NPR 1467	50% OFF	4.1	Very Good
OYO 222 Hotel New Himalayan	Mitranagar, New Buspark, Kathmandu	800	NPR 2077	61% OFF	3.9	Good
OYO 260 Hotel Cultural Inn	Sinamangal, Manpowerbazar, Kathmandu	800	NPR 2970	73% OFF	4.6	Excellent
OYO 161 Ram Janaki Hotel	Sambhu Marg, Airport-9, Kathmandu	800	NPR 2039	61% OFF	4.7	Excellent
OYO 233 Waling Fulbari Guest House	Mitranagar-26, Gondhuli Marg, New Buspark, Kathmandu	800	NPR 3169	75% OFF	4.6	Excellent
OYO 171 Hotel Fewa Darshan	Sora Khutte, Kathmandu	800	NPR 2011	60% OFF	3.2	Fair
OYO 265 Hotel Black Stone	Bhattedanda Devistan Rd, Kavre Nitya Chandeshwor, Kathmandu	833	NPR 2178	62% OFF	0	0
OYO 148 hotel green orchid	paknajol road , thamel , Kathmandu	833	NPR 2277	63% OFF	4.1	Very Good
OYO 146 Somewhere Hotel & Resturant	Opposite to Airport Gate, Beside Global IME Bank, Kalimatidol, Kathmandu	833	NPR 1667	50% OFF	3.9	Good
OYO 297 Hotel Aayam	Ganbihar Galli, Bagdurbar, Sundhara, Kathmandu	856	NPR 1713	50% OFF	0	0
OYO 160 Hotel Shraddha Palace	Gaushala chwok near by Pashupatinath Temple, Kathmandu	867	NPR 2578	66% OFF	4.5	Excellent
OYO 247 Hotel North Pole Inn	Araniko Highway, Kathmandu	867	NPR 2701	68% OFF	4.5	Excellent
OYO 232 Hotel Titanic Pvt Ltd	Mhepi Marg-29, Kathmandu	867	NPR 2695	68% OFF	3.9	Good
OYO 267 Hotel Tanahun Vyas	à¤à¥à¤à¤à¤¬à¥à¤ à¤à¤à¥à¤°à¤ªà¤¥ à¤à¥à¤°à¤¾à¤ à¤à¤¤à¥à¤¤à¤°, Kathmandu	867	NPR 2191	60% OFF	3.8	Good
OYO 264 Hote Antique Kutty	Amrit Marg, Kathmandu	867	NPR 1734	50% OFF	0	0
OYO 144 Hotel Zhonghau	Kathmandu, Satghumti, Kathmandu	867	NPR 1734	50% OFF	4.1	Very Good
OYO 175 Hotel Felicity	Amrit Marg, Kathmandu	867	NPR 2041	58% OFF	3.9	Good
OYO 273 Hotel Rara Palace	Mitranagar-26 , Naya Buspark, Kathmandu	880	NPR 1761	50% OFF	0	0
OYO 210 Andes Hotel Pvt Ltd	Ward No16, Paknajol Marg, Kathmandu	900	NPR 2604	65% OFF	3.8	Good
OYO 135 Lost Garden Apartment and Guest House	Lazimpat, Kathmandu, Kathmandu	900	NPR 1800	50% OFF	4.2	Very Good
OYO 135 Lost Garden Apartment and Guest House	Lazimpat, Kathmandu, Kathmandu	900	NPR 1800	50% OFF	4.2	Very Good
OYO 281 Twins Kitchen	B.P.Highway, Hurkha, Dhulikhel, Kathmandu	917	NPR 2933	69% OFF	3.9	Good
OYO 166 Kathmandu Tourist Home	Saat Ghumti Marg, Kathmandu	933	NPR 2604	64% OFF	4.1	Very Good
OYO 200 Tibet Peace Guest House	Ward No16, Paknajol Marg, Kathmandu	933	NPR 1867	50% OFF	4.1	Very Good
OYO 236 Hotel Beli Nepal	Dhalko Marga, Kathmandu	933	NPR 2472	62% OFF	3.9	Good
OYO 276 White Orchid Resort	Araniko Highway, Bansghari, Bhaktapur.(On the Way to Ktm Fun Valley), Kathmandu	933	NPR 2510	63% OFF	3.9	Good
OYO 195 Global Hotel And Restaurant	Gaurighat, Kathmandu	933	NPR 1867	50% OFF	0	0
OYO 231 Hotel Magnificent View	Keshar Mahal Marg, Kathmandu	933	NPR 2675	65% OFF	3.6	Good
OYO 266 Hotel Grand Stupa	Tushal Chowk, Boudha, Kathmandu	933	NPR 3370	72% OFF	4.3	Very Good
OYO 293 Royal Bouddha Hotel	Mahankal, Boudhanath-6, Kathmandu	933	NPR 1867	50% OFF	0	0
OYO 169 Hotel Cosmic	Z Street, Thamel, Kathmandu	933	NPR 3080	70% OFF	3.8	Good
OYO 105 Hotel Travel INN	GAUSHALA, Kathmandu	960	NPR 4352	78% OFF	4.2	Very Good
OYO 123 Hotel Prince Plaza	Kathmandu, Nepal, Kathmandu 44600, Nepal, Kathmandu	960	NPR 1920	50% OFF	4.5	Excellent
OYO 158 Hotel Premium	Paknajol Rd, Kathmandu, Kathmandu	967	NPR 1934	50% OFF	4.1	Very Good
OYO 287 Hotel Buddha Palace	Mahadev Marg, Kathmandu	1000	NPR 2000	50% OFF	4.3	Very Good
OYO 252 Hotel Black Stone	Birendra Aishwarya Marg, Kathmandu	1000	NPR 3459	71% OFF	4.2	Very Good
OYO 11457 Kathmandu Resort Hotel	Sagarmatha Bazar, Kathmandu	1000	NPR 2000	50% OFF	4.2	Very Good
OYO Home 167 Adventure	Bhagwan Bahal, Amrit Marg Thamel - 26, Kathmandu	1000	NPR 2513	60% OFF	4.4	Very Good
OYO 137 Hotel Pranisha Inn	Shambhu marg, Road no. 2 Airport Road, Kathmandu	1000	NPR 2000	50% OFF	4.1	Very Good
OYO 256 Mount Princess Hotel	Araniko Highway, Kathmandu	1000	NPR 2644	62% OFF	3.5	Good
OYO 295 Asha Lodges	Araniko Highway, Dhulikhel, Kathmandu	1000	NPR 2000	50% OFF	0	0
OYO 104 Hotel Baltic INN	Tribhuvan International Airport, Sinamangal, Kathmandu	1000	NPR 2714	63% OFF	4	Very Good
OYO 209 Tibet Peace Inn	Paknajol Marg, Kathmandu	1000	NPR 2000	50% OFF	4.5	Excellent
OYO 155 Sankata Hotel & Apartment	Chhusya Galli, thamel, Kathmandu	1067	NPR 2134	50% OFF	4.2	Very Good
OYO 275 Sunshine Garden Resort	Bhatey Dhikor-9 Bhakatapur, Kathmandu	1067	NPR 2134	50% OFF	3.7	Good
OYO 174 Shiva Shankar Hotel	Pashupati Nath Road, Kathmandu, Kathmandu	1067	NPR 3520	70% OFF	4.2	Very Good
OYO 280 Hob Nob Garden Resort	Jagati Bridge, Araniko Highway, Bhaktapur, Kathmandu	1067	NPR 2134	50% OFF	0	0
OYO 120 Hotel Tayoma	J P Road, Thamel, Kathmandu	1120	NPR 2240	50% OFF	3.8	Good
OYO 238 Mustang Thakali Kitchen And Guest House	Boudha 6 - Jorpati Rd, Kathmandu	1133	NPR 2670	58% OFF	4	Very Good
OYO 255 White Zambala Hotel	Dharatol, Boudha-6, Kathmandu	1133	NPR 2670	58% OFF	4.2	Very Good
OYO 11459 Osho Holiday INN	Amrit Marg, Kathmandu 44600, Nepal, Kathmandu	1133	NPR 2267	50% OFF	3.5	Good
OYO 206 Mount View Homes	Boudhha, Kathmandu	1167	NPR 2334	50% OFF	4.7	Excellent
OYO 11468 Kathmandu Embassy Hotel	Surya Mukhi Marga, Kathmandu 44600, Nepal, Kathmandu	1200	NPR 3517	66% OFF	4	Very Good
OYO 153 Aster Hotel Nepal	Bharma Kumari Margh, Jyatha, Thamel, Kathmandu	1200	NPR 3666	67% OFF	4.5	Excellent
OYO 219 Hotel Royal Kusum	pingalasthan, 8, à¤à¤¾à¤ à¤®à¤¾à¤¡à¥à¤, Kathmandu	1200	NPR 2400	50% OFF	3.9	Good
OYO 136 Hotel Om Plaza	Airport, Tilganga, Kathmandu	1200	NPR 3793	68% OFF	4.2	Very Good
OYO 208 Mount Gurkha Palace	sundhara, Bagdurbar, peaceline #44, Kathmandu	1200	NPR 2827	58% OFF	3.4	Fair
OYO 207 Hotel Cirrus	Karvepalanchok, Kathmandu	1200	NPR 2971	60% OFF	3.7	Good
OYO 246 Tensar Hotel	Tusal Marg 6, Boudha, Kathmandu	1233	NPR 2467	50% OFF	4.4	Very Good
OYO 156 Hotel Sweet Town	Thamel Bhagawati Marg, Kathmandu	1440	NPR 3534	59% OFF	4.3	Very Good
OYO 279 City Gaon Resort	Bachamari, Nankhel-9, Araniko Highway, Kathmandu	1467	NPR 3517	58% OFF	0	0
OYO 131 Hotel Milarepa	Thamel, Kathmandu, Kathmandu	1467	NPR 3707	60% OFF	4.4	Very Good
OYO 11474 Gangaur Regency Boutique Hotel	Durbarmarg, Kathmandu	1533	NPR 3067	50% OFF	4.4	Very Good
OYO 234 Nepal Tara	Narsingh Chowk Marg, Kathmandu, Kathmandu	1600	NPR 3927	59% OFF	4.3	Very Good
OYO 159 Hotel Highlander	Thamel, Kathmandu	1600	NPR 3927	59% OFF	3.8	Good
OYO 172 Hotel Deepshree	Airport , Sinamangal , Kathmandu	1734	NPR 3334	48% OFF	4.5	Excellent
OYO 152 Kathmandu Airport Hotel	Ring Road Near Kathmandu Airport, Kathmandu	3360	NPR 4960	32% OFF	4.8	Excellent
OYO 278 Hotel View Bhaktapur	Araniko Highway, Suryabinayak, Kathmandu	4400	NPR 7333	40% OFF	0	0
OYO 217 Shiva Tirupati Hotel	Ward No. 29, Putali Sadak, Kathmandu, Kathmandu	6400	NPR 9427	32% OFF	0	0
OYO 139 Hotel Sujata Inn	shambhu marga , airport road., Kathmandu	65067	0	0	0	0
OYO 199 Hotel Access Nepal	Thamel, Kathmandu	81734	0	0	4.8	Excellent






