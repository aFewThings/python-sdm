import torch
import torch.optim as optimizer
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from lib.cnn.predict import predict
from lib.evaluation import evaluate
from lib.utils import save_model_state
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple


def fit(model, train, validation, device, loss=torch.nn.CrossEntropyLoss(), 
        iterations=(90, 130, 150, 170, 180), log_modulo=500, val_modulo=5,
        lr=0.1, weight_decay=1e-4, gamma=0.1,  momentum=0.9, batch_size=124,  n_workers=8,  validation_size=-1,
        metrics=(ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationAccuracyMultiple([1, 10, 30])),
        save_model_path='./pretrained/model.pt'):
    """
    This function performs a model training procedure.
    :param model: the model that should be trained
    :param train: The training set
    :param validation: The validation set
    :param iterations: This is a list of epochs numbers, it indicates when to changes learning rate
    :param log_modulo: Indicates after how many batches the loss is printed
    :param val_modulo: Indicates after how many epochs should be done a validation
    :param lr: The learning rate
    :param gamma: The coefficient to apply when decreasing the learning rate
    :param momentum: The momentum
    :param batch_size: The mini batch size
    :param n_workers: The number of parallel job for input preparation
    :param validation_size: The maximum number of occurrences to use in validation
    :param metrics: The list of evaluation metrics to use in validation
    """

    # training parameters
    max_iterations = iterations[-1]

    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    print('Training...')

    #opt = optimizer.SGD(model.parameters(), lr=lr, momentum=momentum)
    opt = optimizer.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = MultiStepLR(opt, milestones=list(iterations), gamma=gamma)

    # number of batches (steps or iterations of one epoch) in the ml
    epoch_size = len(train_loader)

    # one log per epoch if value is -1
    log_modulo = epoch_size if log_modulo == -1 else log_modulo

    best_val = 0
    best_res = str()

    for epoch in range(max_iterations):
        model.train()

        # printing new epoch
        print('-' * 5 + ' Epoch ' + str(epoch + 1) + '/' + str(max_iterations) +
              ' (lr: ' + str(scheduler.get_last_lr()) + ') ' + '-' * 5)

        running_loss = 0.0

        for idx, data in enumerate(tqdm(train_loader)):

            # get the inputs
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            train_loss = loss(outputs, labels)

            # zero the parameter gradients
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # print loss
            running_loss += train_loss.item()
            if idx % log_modulo == log_modulo - 1:  # print every log_modulo mini-batches
                tqdm.write('[%d, %5d] train loss: %.5f' % (epoch + 1, idx + 1, running_loss / log_modulo))
                #print('[%d, %5d] loss: %.5f' % (epoch + 1, idx + 1, running_loss / log_modulo))
                running_loss = 0.0

        # end of epoch update of learning rate scheduler
        scheduler.step()

        # validation of the model
        if epoch % val_modulo == val_modulo - 1:
            validation_id = str(int((epoch + 1) / val_modulo))

            # train loss
            predictions, labels = predict(
                model, train, device, batch_size=batch_size, validation_size=validation_size, n_workers=n_workers
            )
            preds, lbs = torch.from_numpy(predictions), torch.from_numpy(labels)
            train_loss = loss(preds, lbs)
            print('[validation_id: %s] total train loss: %.5f' % (validation_id, train_loss.item()))

            # validation loss
            predictions, labels = predict(
                model, validation, device, batch_size=batch_size, validation_size=validation_size, n_workers=n_workers
            )
            preds, lbs = torch.from_numpy(predictions), torch.from_numpy(labels)
            val_loss = loss(preds, lbs)
            print('[validation_id: %s] total valid loss: %.5f' % (validation_id, val_loss.item()))

            # evaluate
            res = '\n[validation_id: ' + validation_id + ']\n' + evaluate(predictions, labels, metrics)
            print(res)

            val = metrics[0].auc
            # acc = metrics[0].acc
            if val > best_val:
                best_val = val
                best_res = res
                save_model_state(model, file_path=save_model_path, validation_id="best")

            # save model state dict
            # this is not a checkpoint, the model saved can be load but not to restart learning from there
            # this is for the model selection at the end of training for the final evaluation
            save_model_state(model, file_path=save_model_path, validation_id=str(validation_id))

    # save_model_state(model, file_path=save_model_path, validation_id="final")
    # final validation
    # print('Final validation: ')

    # predictions, labels = predict(
    #     model, validation, device, batch_size=batch_size, validation_size=-1, n_workers=n_workers
    # )

    # res = evaluate(predictions, labels, metrics, final=True)
    # print(res)

    print('Best results: ')
    print(best_res)

    return best_res
