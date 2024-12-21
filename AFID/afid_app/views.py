from django.shortcuts import render
from .ml.prediction import load_and_prepare_data, create_dataset, build_and_train_model, predict_next_30_days, plot_results

def predict_view(request):
    if request.method == 'POST' and request.FILES['dataset']:
        dataset = request.FILES['dataset']
        filepath = f"{settings.MEDIA_ROOT}/{dataset.name}"

        # Save the uploaded file
        with open('media\plots', 'wb+') as destination:
            for chunk in dataset.chunks():
                destination.write(chunk)

        # Use functions from prediction.py
        time_step = 100
        dates, normalized_prices, scaler = load_and_prepare_data(filepath)

        training_size = int(len(normalized_prices) * 0.65)
        train_data = normalized_prices[0:training_size, :]
        test_data = normalized_prices[training_size:len(normalized_prices), :]

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = build_and_train_model(X_train, y_train, X_test, y_test)
        future_predictions = predict_next_30_days(model, scaler, test_data, time_step)

        return render(request, 'result.html', {'predictions': future_predictions.tolist()})

    return render(request, 'predict_form.html')


from django.shortcuts import render
from django.http import HttpResponse
from .forms import DatasetUploadForm
from .utils import process_dataset_and_predict  # Assuming you have a utility function for prediction

def predict_form(request):
    """
    View to handle the dataset upload form.
    Displays the form and handles POST requests for file upload and predictions.
    """
    if request.method == "POST":
        # Instantiate the form with data and files from the request
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded file
            dataset = request.FILES['dataset']

            # Call a utility function to process the dataset and generate predictions
            predictions = process_dataset_and_predict(dataset)

            # Render the result template with predictions
            return render(request, 'result.html', {'predictions': predictions})
    else:
        # If it's a GET request, show the empty form
        form = DatasetUploadForm()

    # Render the template with the form to upload dataset
    return render(request, 'predict_form.html', {'form': form})

def result_view(request):
    """
    Optionally handle a view to display predictions.
    If you need a separate page for results, use this.
    Otherwise, the result can be handled directly in the `predict_form` view.
    """
    return render(request, 'result.html')
