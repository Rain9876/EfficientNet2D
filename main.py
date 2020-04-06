from Training.Fine_tuning import *

# md = pre_trained_model()
# # img = imageShow("../data/img.jpg")
# # subTestForPreTraining(md, img)
# fine_tuning(md, 10)
#

# print(sys.path)

md = pre_trained_model()
model = fine_tuning(md, 100)





#
# from skorch import NeuralNetClassifier
#
# net = NeuralNetClassifier(
#     model,
#     max_epochs=10,
#     lr=0.1,
#     # Shuffle training data on each epoch
#     iterator_train__shuffle=True,
# )
#
# from sklearn.model_selection import GridSearchCV
#
# params = {
#     'lr': [0.01, 0.02],
#     'max_epochs': [10, 20],
#     'module__num_units': [10, 20],
# }
#
# gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')
#
# train_loader, test_loader = ImageProcessing()
#
# # (inputs, targets) =  train_loader.dataset
#
# from skorch import NeuralNet
# nNet = NeuralNet(model, criterion=torch.nn.MSELoss)
# nNet.fit(train_loader.dataset)
# print(gs.best_score_, gs.best_params_)
#
#

