from joblib import load
import pandas as pd
from django.shortcuts import render

# Charger le modèle et le pipeline de prétraitement
model = load('./saved_models/trained_pipe_bis.joblib')

def predictor(request):
    if request.method == 'POST':
        # Obtenir les données d'entrée de la requête
        sqft_living = int(request.POST['sqft_living'])
        sqft_lot = int(request.POST['sqft_lot'])
        # sqft_living15 = int(request.POST['sqft_living15'])
        # sqft_lot15 = int(request.POST['sqft_lot15'])
        sqft_above = int(request.POST['sqft_above'])
        sqft_basement = int(request.POST['sqft_basement'])
        floors = float(request.POST['floors'])
        bedrooms = int(request.POST['bedrooms'])
        bathrooms = float(request.POST['bathrooms'])
        view = int(request.POST['view'])
        grade = int(request.POST['grade'])
        condition = int(request.POST['condition'])
        if 'waterfront' in request.POST:
            waterfront = int(request.POST['waterfront'])
        else:
            waterfront = 0
        zipcode = int(request.POST['zipcode'])
        lat = float(request.POST['lat'])
        long = float(request.POST['long'])
        yr_built = int(request.POST['yr_built'])
        yr_renovated = int(request.POST['yr_renovated'])

        # Convertir les données en un tableau NumPy
        X = pd.DataFrame({'sqft_living':[sqft_living],
                          'sqft_lot':[sqft_lot],
                        #   'sqft_living15': [sqft_living15], 
                        #   'sqft_lot15': [sqft_lot15],
                          'sqft_above': [sqft_above],
                          'sqft_basement' : [sqft_basement], 
                          'floors': [floors],
                          'bedrooms': [bedrooms],
                          'bathrooms' : [bathrooms],
                          'view' : [view],
                          'grade' : [grade],
                          'condition' : [condition],
                          'waterfront' : [waterfront],
                          'zipcode' : [zipcode],
                          'lat' : [lat],
                          'long' : [long],
                          'yr_built' : [yr_built],
                          'yr_renovated': [yr_renovated]})

        # Faire la prédiction
        y_pred = model.predict(X)
        # Arrondir la valeur à un entier
        rounded_y_pred = round(y_pred[0])

        # Formater la chaîne de caractères
        formatted_y_pred = '{:,.0f}'.format(rounded_y_pred).replace(',', ' ')
        print(formatted_y_pred)

        # Retourner la réponse au client
        return render(request, 'main.html', {'result' : formatted_y_pred})
    return render(request, 'main.html')
