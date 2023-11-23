

def get_data(filepath):
    current_path = os.getcwd()
    df = pd.read_csv(current_path+filepath)
    return df


def merge_data_by_country(df_nomadlist, df_SDG):
    filtered_country_df_nomadlist = df_nomadlist['country'].unique()
    country_codes = {find_countries(country)[0][0].alpha_3: country for country in filtered_country_df_nomadlist if find_countries(country)}

    #Skim the SDG dataset
    for code in country_codes.keys():
        df_SDG.loc[df_SDG['Country Code'] == code, 'Country Name'] = country_codes[code]

    df_SDG = df_SDG.loc[df_SDG['Country Name'].isin(filtered_country_df_nomadlist)]
    number_of_countries = len(df_SDG['Country Code'].unique())

    #Finding columns that preserve all data, i.e. that are shared by every city
    df_SDG_count = df_SDG.groupby(['Indicator Name']).count()
    common_indicators = df_SDG_count[df_SDG_count['2015'] == number_of_countries].reset_index()['Indicator Name']
    df_SDG = df_SDG.loc[df_SDG['Indicator Name'].isin(common_indicators)]


    #Actually merge the data
    df_SDG = df_SDG[['Country Name', 'Indicator Name', '2015']]
    df_SDG = df_SDG.pivot(index = 'Country Name', columns = 'Indicator Name', values = '2015')

    df_SDG = df_SDG.reset_index()
    df_SDG_cleaned = df_SDG.rename(columns={'Country Name': 'country'})

    df_nomadlist = df_nomadlist.merge(df_SDG_cleaned, how='outer', on='country')
    return df_nomadlist
#Remove data non matching with our other dataset

##EXTRACTING COUNTRY CODES TO MATCH COUNTRIES FROM SDG AND NOMADLIST
##SINCE THE SAME COUNTRY IS OFTEN CALLED IN 2 DIFFERENT WAYS"

##THEN, MERGING THE DATASETS



'''
df_SDG_count = df_SDG.groupby(['Country Code']).count()
min_indicators_cc = df_SDG_count['2015'].idxmin()
indicators_list = df_SDG[df_SDG['Country Code'] == min_indicators_cc]['Indicator Name']
'''


#Approximately 2 cities per user, and 2 records per city
#We could use this to build user profiles maybe
'''
filepath_users = "/datasets/archive/Travel_details_dataset.csv"
df_users = pd.read_csv(current_path+filepath_users)
print(df_users.columns)
print(len(df_users))
df_users_unique_cities = df_users['Destination'].unique()
print(len(df_users_unique_cities))
df_users_unique_users = df_users['Traveler name'].unique()
print(len(df_users_unique_users))
'''


##START OF FLICKR PART
def configure():
    load_dotenv()
#FINDING CITIES COORDINATES

def get_city_name_from_nomadlist(df_nomadlist):
    cities = df_nomadlist['place_slug']
    countries = df_nomadlist['country']
    cities_list = []
    for city, country in zip(cities, countries):
        number_of_country_words = len(country.split(' '))
        city_name = ' '.join(city.split('-')[:-number_of_country_words])
        cities_list.append(city_name)

    cities_list = sorted(cities_list)
    return cities_list

def get_geoloc_per_city(cities_list, saving_filepath):
    current_path = os.getcwd()
    geolocator = Nominatim(user_agent='myapplication')
    cities_coord = {}

    if os.path.exists(current_path+saving_filepath) and os.path.getsize(current_path+saving_filepath) > 0:
        f = open(current_path+saving_filepath)
        cities_coord = json.load(f)
    else:
        for city in tqdm(cities_list):
            location = geolocator.geocode(city)
            lat = location.raw['lat']
            lon = location.raw['lon']
            cities_coord[city] = (lat, lon)

        with open('cities_coord.json', 'w') as fp:
            json.dump(cities_coord, fp)
    return cities_coord

#USING CITIES COORDINATES TO GET ALL TRAVEL IMAGES IN FLICKR TAKEN IN THOSE CITIES
#BUILDING DICTIONARY OF USER AND CITIES HE TRAVELLED TO


def get_flickr_data(cities_coord, saving_filepath):
    current_path = os.getcwd()
    flickr = flickrapi.FlickrAPI(os.getenv('api_key'), os.getenv('api_secret'), format='parsed-json')


    extras = ['description','tags','url_sq', 'url_t', 'url_s', 'url_q', 'url_m', 'url_n', 'url_z', 'url_c', 'url_l', 'url_o']
    users = {}
    if os.path.exists(current_path+saving_filepath) and os.path.getsize(current_path+saving_filepath) > 0:
        f = open(current_path+saving_filepath)
        users = json.load(f)
    else:
        for city, coords in tqdm(cities_coord.items()): 
            try:
                images = flickr.photos.search(text='travel', lat = coords[0], lon = coords[1], radius = '30', radius_units = 'km', extras=extras)
                for image in images['photos']['photo']:
                    user = image['owner']
                    if user in users.keys():
                        users[user].append(city)
                    else:
                        users[user] = [city]
            except:
                print(f'Images for {city} not found or another error encountered')

            with open(saving_filepath, 'w') as fp:
                json.dump(users, fp)
    return users


def process_data(users):
    users_cities_rating = {}
    number_of_values = 0
    for key, value in users.items():
        uniques = set(value)
        avg_pics_per_place = len(value)/len(uniques)
        city_count = Counter(value)
        #Normalization
        for item, count in city_count.items():
            city_count[item] /= avg_pics_per_place

        users_cities_rating[key] = city_count

        users[key] = uniques
        number_of_values += len(users[key])

    return users, users_cities_rating

def matrix_factorization(matrix, size=8, steps=50, eta = 0.001, lambd = 0.01, threshold = 0.01):
    user_len = len(matrix[:, 0])
    item_len = len(matrix[0,:])

    decomp_rows = np.random.rand(user_len,size)
    decomp_cols = np.random.rand(item_len, size)
    #
    decomp_cols = decomp_cols.T

    for step in range(steps):
        for i in tqdm(range(len(matrix))):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    err_ij = matrix[i][j] -np.dot(decomp_rows[i,:], decomp_cols[:,j])

                    for k in range(size):
                        decomp_rows[i][k] = decomp_rows[i][k] + eta * (2*err_ij*decomp_cols[k][j] - lambd * decomp_rows[i][k])
                        decomp_cols[k][j] = decomp_cols[k][j] + eta * (2*err_ij*decomp_rows[i][k] - lambd * decomp_cols[k][j])

        err = np.dot(decomp_rows, decomp_cols)
        e = 0

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j]!=0:
                    e = e + pow(matrix[i][j] - np.dot(decomp_rows[i,:], decomp_cols[:,j]) ,2)

        print(f"Error at step {step} is {e}")
        if e < threshold:
            break

    return decomp_rows, decomp_cols.T

##Matrix Factorization

def main():
    configure()
    filepath_nomadlist = '/datasets/nomadlist/cities_predict.csv'
    df_nomadlist = get_data(filepath_nomadlist)

    ##IMPORTING SDG DATA
    filepath_SDG = "/datasets/SDG_CSV/SDGData.csv"
    df_SDG = get_data(filepath_SDG)
    df_SDG = df_SDG[['Country Code', 'Country Name', 'Indicator Name', '2015']]
    df_SDG = df_SDG.dropna()

    cities_list = get_city_name_from_nomadlist(df_nomadlist)
    df_nomadlist = merge_data_by_country(df_nomadlist, df_SDG)

    #Flickr part
    saving_filepath = '/cities_coord.json'
    cities_coord = get_geoloc_per_city(cities_list, saving_filepath)
    saving_filepath = '/users.json'
    users = get_flickr_data(cities_coord, saving_filepath)
    users, users_cities_rating = process_data(users)


    #Create the matrices to factorize
    matrix_to_factorize_0_1 = np.zeros((len(users), len(cities_list)), dtype='bool')
    matrix_to_factorize_rated = np.zeros((len(users), len(cities_list)), dtype='int')

    #Populate them
    for i,city in enumerate(cities_list):
        for j,(user,city_values) in enumerate(users.items()):
            if city in city_values:
                matrix_to_factorize_0_1[j][i] = 1
                matrix_to_factorize_rated[j][i] = users_cities_rating[user][city]

    decomp_users, decomp_cities = matrix_factorization(matrix_to_factorize_0_1)



if __name__=='__main__':   
    main()