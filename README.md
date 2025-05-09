# NomadListRecommender

The folder 'datasets' contains all the datasets used in the project:
- sustainable development goals (SGDs dataset): set of 17 goals to achieve a more sustainable and equitable world by the years 2030. It is interesting because it includes for almost all countries in the world information on various indicators such as poverty, hunger, health, education, gender equality, clean water, sanitation, affordable and clean energy, economic growth, industry, innovation, reduced inequality, sustainable cities, responsible consumption, climate action, life below water, life on land, peace, justice, and partnerships.
- flicks: json files of information extracted by flicks platform (see code in *NomadListRecommender.ipynb*)
- nomadlist: main dataset used in this project which contains information regarding 730 cities provided by digital nomads

The notebook **NomadListRecommender.ipynb** contains all the code used to build the recommendation system and specifically it comprises:
- retrieving data
- data processing and EDA
- methodologies which comprends the implementation of clustering, similar items and minhashing tecniques 
- user recommendations which comprends the implementation of collaborative filtering and matrix factorization

# How to run
To run the notebook, just download the folder and run *NomadListRecommender.ipynb*. 
If you wish to download again flickr data, delete the content of all jsons files in /datasets/flickr and run the code again. Pls note that you also must have an API key and a secret provided by flickr stored in a .env file.
