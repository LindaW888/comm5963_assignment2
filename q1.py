import pandas as pd
import sklearn
import torch
import os
import requests

PREDICTOR_COLUMNS = ['Petal_width', 'Petal_length', 'Sepal_width', 'Sepal_length']
TARGET_COLUMN = 'Species_No'
TARGET_CLASS_DICT = {0: 'Setosa', 1: 'Versicolor', 2: 'Verginica'}


def run_prediction(_model, x_tensor):
    with torch.no_grad():
        test_predict = _model(x_tensor)
        _, predicted_classes = torch.max(test_predict, 1)
    return predicted_classes

def calc_accuracy(_model, x_tensor, y_tensor):
    predicted_classes = run_prediction(_model, x_tensor)
    # Accuracy = Number of correct prediction / Number of items to be predicted
    return (predicted_classes == y_tensor).sum().item() / y_tensor.size(0)

def read_dataframe() -> pd.DataFrame:
    file_name = 'Iris.xls'
    if os.path.exists(file_name):
        print(f'Reusing previously downloaded file: {file_name}')
    else:
        # Data Reference: https://doi.org/10.7910/DVN/R2RGXR
        # Columns: ['Species_No', 'Petal_width', 'Petal_length', 'Sepal_width', 'Sepal_length', 'Species_name']
        url_iris = 'http://faculty.smu.edu/tfomby/eco5385_eco6380/data/Iris.xls'
        print(f'File did not exists, downloading: {url_iris}')
        with open(file_name, 'wb') as file_obj:
            file_obj.write(requests.get(url_iris).content)
    return pd.read_excel(file_name)

def load_tensors():
    df_iris = read_dataframe()
    # Adjust the species_no to start from 0
    df_iris['Species_No'] = df_iris['Species_No'] - 1
    # Train-Test split
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
        df_iris[PREDICTOR_COLUMNS], df_iris[TARGET_COLUMN], test_size=0.2, random_state=5963)
    # Convert to PyTorch tensors
    _train_x_tensor, _test_x_tensor = torch.FloatTensor(train_x.to_numpy()), torch.FloatTensor(test_x.to_numpy())
    _train_y_tensor, _test_y_tensor = torch.LongTensor(train_y.to_numpy()), torch.LongTensor(test_y.to_numpy())
    return _train_x_tensor, _test_x_tensor, _train_y_tensor, _test_y_tensor

def train_model(x_tensor, y_tensor) -> torch.nn.Sequential:
    torch.manual_seed(5963)
    # Configurable: Your model structure
    model = torch.nn.Sequential(
        torch.nn.Linear(len(PREDICTOR_COLUMNS), 8),  # Input 4 predictors, Output 8 neurons
        torch.nn.Linear(8, len(TARGET_CLASS_DICT))   # Input 8 neurons, Output 3 classes (Setosa, Versicolor, Verginica)
    )
    # Cross Entropy Loss is used for classification
    loss_function = torch.nn.CrossEntropyLoss()

    # Configurable: Hyper-parameters
    num_epochs = 10
    learning_rate = 0.8
    # Configurable: optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Start training
    for epoch in range(num_epochs):
        optimizer.zero_grad() # Resets gradients
        train_predict = model(x_tensor)  # Make a prediction
        loss = loss_function(train_predict, y_tensor)  # Calculate loss
        loss.backward()  # Calculate gradient
        optimizer.step()  # Update weights using the graident
        if (epoch + 1) % (num_epochs/10) == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model



if __name__ == '__main__':
    train_x_tensor, test_x_tensor, train_y_tensor, test_y_tensor = load_tensors()
    your_model = train_model(train_x_tensor, train_y_tensor)
    print(f'Train accuracy: {calc_accuracy(your_model, train_x_tensor, train_y_tensor):.2%}')
    print(f'Test accuracy: {calc_accuracy(your_model, test_x_tensor, test_y_tensor):.2%}')
    torch.save(your_model, 'q1.pth')
    print('Exported model as file')
    guess = [2, 5, 3, 6]
    predict_species = TARGET_CLASS_DICT[run_prediction(your_model, torch.Tensor(guess).view(1, -1))[0].item()]
    msg = 'CORRECT' if predict_species == 'Verginica' else 'WRONG'
    print(f'[{msg}] Prediction of {dict(zip(PREDICTOR_COLUMNS, guess))} from your model: {predict_species}')