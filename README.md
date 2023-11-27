# NomadListRecommender

The folder 'datasets' contains all the datasets used in the project:
- sustainable development goals (SGDs dataset): set of 17 goals to achieve a more sustainable and equitable world by the years 2030. It is interesting because it includes for almost all countries in the world information on various indicators such as poverty, hunger, health, education, gender equality, clean water, sanitation, affordable and clean energy, economic growth, industry, innovation, reduced inequality, sustainable cities, responsible consumption, climate action, life below water, life on land, peace, justice, and partnerships.
- flicks: json files of information extracted by flicks platform (see code in *NomadListRecommender.ipynb*)
- nomadlist: main dataset we used in the project which contains information regarding 730 cities provided by digital nomads

The notebook **NomadListRecommender.ipynb** contains all the code of the recommendation system we have built and specifically it comprises:
- retriving data
- data processing and EDA
- methodologies which comprends teh code to run clustering, similar items and minhashing
- user recommendations which comprends collaborative filtering and matrix factorization
