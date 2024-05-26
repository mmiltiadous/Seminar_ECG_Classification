import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, ReLU, Dropout, Flatten, Dense, Add, Concatenate, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold
import time
import random



def create_time_series_classifier(input_shape, num_classes, dropout_rate=0.3):
    inputs = Input(shape=input_shape)
    
    #first layer separately to avoid skip connections
    x = Conv1D(64, 15, padding='same',activation='linear', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    skip_connections = [x]
    
    for i in range(15):
        x = Conv1D(64, 15, padding='same', activation='linear', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)
        
        if i % 2 == 0 and len(skip_connections) > 1:  
            skip = skip_connections[-2]
            if skip.shape[1] != x.shape[1]:  # If not compatible shapes
                skip = Conv1D(64, 1, padding='same')(skip)
                if skip.shape[1] > 1:  
                    skip = MaxPooling1D(pool_size=2)(skip)
            x = Add()([x, skip]) 
        
        if x.shape[1] > 1:  
            x = MaxPooling1D(pool_size=2)(x)
        skip_connections.append(x)
    
    x = GlobalAveragePooling1D()(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def load_data(data_dir, data_name):
    train_data = pd.read_csv(data_dir + data_name + f'/{data_name}_train.csv', header=None)
    val_data = pd.read_csv(data_dir + data_name + f'/{data_name}_val.csv', header=None)
    test_data = pd.read_csv(data_dir + data_name + f'/{data_name}_test.csv', header=None)

    train_labels = train_data.iloc[:, 0].values
    train_signals = train_data.iloc[:, 1:].values

    val_labels = val_data.iloc[:, 0].values
    val_signals = val_data.iloc[:, 1:].values

    test_labels = test_data.iloc[:, 0].values
    test_signals = test_data.iloc[:, 1:].values

    # Reshape in 3D array format
    train_signals = train_signals.reshape(-1, train_signals.shape[1], 1)
    val_signals = val_signals.reshape(-1, val_signals.shape[1], 1)
    test_signals = test_signals.reshape(-1, test_signals.shape[1], 1)

    num_classes = len(np.unique(train_labels))
    train_labels = train_labels.astype(int)
    val_labels = val_labels.astype(int)
    test_labels = test_labels.astype(int)
    
    return train_signals, train_labels, val_signals, val_labels, test_signals, test_labels


def calculate_metrics(y_true, y_pred, duration):

    res = pd.DataFrame(data=np.zeros((1, 10), dtype=float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 
                                'precision_0', 'recall_0',
                                'precision_1', 'recall_1',
                                'precision_2','recall_2',
                                'duration'])
    
    res['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    for i in range(3):
        y_true_i = [1 if label == i else 0 for label in y_true]
        y_pred_i = [1 if label == i else 0 for label in y_pred]
        
        res[f'precision_{i}'] = precision_score(y_true_i, y_pred_i, zero_division=0)
        res[f'recall_{i}'] = recall_score(y_true_i, y_pred_i, zero_division=0)

    clas_rep = classification_report(y_true, y_pred, target_names=['class 0', 'class 1', 'class 2'], zero_division=0)
    
    res['duration'] = duration


    #Accuracies using conf matrix
    #change
    res2 = pd.DataFrame(data=np.zeros((1, 3), dtype=float), index=[0],
                       columns=['accuracy_0', 'accuracy_1', 'accuracy_2'])

    confusion_matrix1 = confusion_matrix(y_true, y_pred)

    num_classes = confusion_matrix1.shape[0]

    # Initialize an array to hold the accuracy for each class
    class_accuracies = np.zeros(num_classes)

    # Calculate the accuracy for each class
    for i in range(num_classes):
        TP = confusion_matrix1[i, i]
        FN = np.sum(confusion_matrix1[i, : ]) - TP
        FP = np.sum(confusion_matrix1[:, i]) - TP
        TN = np.sum(confusion_matrix1) - (TP + FN + FP)
        
        class_accuracies[i] = (TP + TN) / (TP + TN + FP + FN)
    
    for i in range(num_classes):
        res2[f'accuracy_{i}'] = class_accuracies[i]

    return res, clas_rep, res2 #change

######
######
#NO CROSS VALIDATION
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

data_dir = 'DATA_SADL/'
data_name = 'mydata18287_noother'
output_dir = f'results_cnn/{data_name}/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


num_iterations = 5
accuracies = []
precisions = []
recalls = []

for i in range(num_iterations):

    print(f"Iteration {i+1}")
        
    train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = load_data(data_dir, data_name)

    input_shape = train_signals.shape[1:]  
    num_classes = len(np.unique(train_labels))

    model = create_time_series_classifier(input_shape, num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

    checkpoint_path = os.path.join(output_dir, f'best_model_iter{i+1}.h5')
    
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    start_time = time.time()
    
    hist = model.fit(train_signals, train_labels, epochs=1500, batch_size=64, 
                     validation_data=(val_signals, val_labels), callbacks=[early_stopping, model_checkpoint])
    duration = time.time() - start_time

    print('Fit time:', duration)

    model.load_weights(checkpoint_path)

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_signals, test_labels)
    print(f"Iteration {i+1} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")



    y_pred = model.predict(test_signals, batch_size = 64)
    y_pred_classes = np.argmax(y_pred, axis=1)

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_dir + f'history_iter{i+1}.csv', index=False)
    df_metrics, class_rep, df_metrics2 = calculate_metrics(test_labels, y_pred_classes, duration) #change

    df_metrics.to_csv(output_dir + f'df_metrics_iter{i+1}.csv', index=False)

    df_metrics2.to_csv(output_dir + f'df_accuracies_iter{i+1}.csv', index=False) #change


    full_path = output_dir + f'classification_report_iter{i+1}.csv'

    with open(full_path, "w") as file:
        file.write(class_rep)


    accuracies.append(df_metrics['accuracy'].values[0])
    precisions.append(df_metrics['precision'].values[0])
    recalls.append(df_metrics['recall'].values[0])

# Calculate mean metrics
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

print("Mean Accuracy:", mean_accuracy)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)

data = {
    'Metric': ['Mean Accuracy', 'Mean Precision', 'Mean Recall'],
    'Value': [mean_accuracy, mean_precision, mean_recall]
}
df = pd.DataFrame(data)

overall_dir = f'results_cnn/{data_name}/overall'  

if not os.path.exists(overall_dir):
    os.makedirs(overall_dir)

file_path = os.path.join(overall_dir, 'overall_metrics.csv')
df.to_csv(file_path, index=False)


######
######
# #CROSS VALIDATION

# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

# # Enable mixed precision
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

# data_dir = 'DATA_SADL/'
# data_name = 'mydata2715'
# output_dir = f'results_cnn/{data_name}/'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = load_data(data_dir, data_name)

# input_shape = train_signals.shape[1:]  
# num_classes = len(np.unique(train_labels))

# num_folds = 5
# accuracies = []
# precisions = []
# recalls = []

# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# for fold, (train_index, val_index) in enumerate(skf.split(train_signals, train_labels)):
#     print(f'Fold {fold + 1}/{num_folds}')
    

#     x_train, x_val = train_signals[train_index], train_signals[val_index]
#     y_train, y_val = train_labels[train_index], train_labels[val_index]

#     # Further split the training into 80-20 train-validation
#     split_index = int(len(x_train) * 0.8)
#     x_train_split, x_val_split = x_train[:split_index], x_train[split_index:]
#     y_train_split, y_val_split = y_train[:split_index], y_train[split_index:]


#     model = create_time_series_classifier(input_shape, num_classes)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

   
#     early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
#     checkpoint_path = os.path.join(output_dir, f'best_model_fold{fold + 1}.h5')
#     model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

   
#     start_time = time.time()
#     hist = model.fit(x_train_split, y_train_split, epochs=1500, batch_size=64, 
#                      validation_data=(x_val_split, y_val_split), callbacks=[early_stopping, model_checkpoint])
#     duration = time.time() - start_time
#     print('Fit time:', duration)

#     # Load best model
#     model.load_weights(checkpoint_path)

#     test_loss, test_accuracy = model.evaluate(test_signals, test_labels)
#     print(f"Fold {fold + 1} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

#     y_pred = model.predict(test_signals, batch_size = 64)
#     y_pred_classes = np.argmax(y_pred, axis=1)

#     hist_df = pd.DataFrame(hist.history)
#     hist_df.to_csv(output_dir + f'history_fold{fold}.csv', index=False)
#     df_metrics, class_rep, df_metrics2 = calculate_metrics(test_labels, y_pred_classes, duration) #change

#     df_metrics.to_csv(output_dir + f'df_metrics_fold{fold}.csv', index=False)

#     df_metrics2.to_csv(output_dir + f'df_accuracies_fold{fold}.csv', index=False) #change


#     full_path = output_dir + f'classification_report_fold{fold}.csv'

#     with open(full_path, "w") as file:
#         file.write(class_rep)

#     accuracies.append(df_metrics['accuracy'].values[0])
#     precisions.append(df_metrics['precision'].values[0])
#     recalls.append(df_metrics['recall'].values[0])

# mean_accuracy = np.mean(accuracies)
# mean_precision = np.mean(precisions)
# mean_recall = np.mean(recalls)

# print("Mean Accuracy:", mean_accuracy)
# print("Mean Precision:", mean_precision)
# print("Mean Recall:", mean_recall)

# data = {
#     'Metric': ['Mean Accuracy', 'Mean Precision', 'Mean Recall'],
#     'Value': [mean_accuracy, mean_precision, mean_recall]
# }
# df = pd.DataFrame(data)

# overall_dir = f'results_cnn/{data_name}/overall'  

# if not os.path.exists(overall_dir):
#     os.makedirs(overall_dir)

# file_path = os.path.join(overall_dir, 'overall_metrics.csv')
# df.to_csv(file_path, index=False)


