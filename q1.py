import sklearn
import torch
from utils import load_train_test_datasets
from static import PREDICTOR_COLUMNS, TARGET_COLUMN, TARGET_CLASS_DICT
from sklearn.model_selection import train_test_split

def run_prediction(_model, x_tensor):
    # Given a model and input, predict the corresponding output
    with torch.no_grad():
        test_predict = _model(x_tensor)
        _, predicted_classes = torch.max(test_predict, 1)
    return predicted_classes

def calc_accuracy(_model, x_tensor, y_tensor):
    predicted_classes = run_prediction(_model, x_tensor)
    # Accuracy = Number of correct prediction / Number of items to be predicted
    return (predicted_classes == y_tensor).sum().item() / y_tensor.size(0)

def load_tensors():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    # Convert to PyTorch tensors
    _train_x_tensor, _test_x_tensor = torch.FloatTensor(train_x.to_numpy()), torch.FloatTensor(test_x.to_numpy())
    _train_y_tensor, _test_y_tensor = torch.LongTensor(train_y.to_numpy()), torch.LongTensor(test_y.to_numpy())
    return _train_x_tensor, _test_x_tensor, _train_y_tensor, _test_y_tensor

def train_model(x_tensor, y_tensor) -> torch.nn.Sequential:
    torch.manual_seed(5963)
    # Configurable: Your model structure
    model = torch.nn.Sequential(
        torch.nn.Linear(len(PREDICTOR_COLUMNS), 8),  # Input 4 predictors, Output 8 neurons
        torch.nn.ReLU(),
        torch.nn.Linear(8, len(TARGET_CLASS_DICT))   # Input 8 neurons, Output 3 classes (Setosa, Versicolor, Verginica)
    )
    # Cross Entropy Loss is used for classification
    loss_function = torch.nn.CrossEntropyLoss()

    # Configurable: Hyper-parameters
    num_epochs = 100
    learning_rate = 0.03
    # Configurable: optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=1)

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