# Liquid neural networks for cancer classification

This project aims to use **liquid neural network** architecture to perform image classification on breast cancer ultrasounds from the [MedMNIST dataset](https://medmnist.com/).

### Background
Ramin Hasani et. al first introduced Liquid Time-constant networks, which allowed networks to have variable non-linear relationships, rather than have predefined fixed non-linearities. See [here](https://doi.org/10.1609/aaai.v35i9.16936). <br>
They then applied these LTC cells within a Neural Circuit Policy (NCP) architecture, which was loosely based off the brain of the C. elegans worm. See [here](https://www.nature.com/articles/s41597-022-01721-8).

The MedMNIST v2 dataset is an opensource dataset of various collections of images for different medical tasks. This project trains off the BreastMNIST subset in order to classify breast ultrasound images as benign/normal (positive) or malignant (negative). <br>
Examples of ultrasound images: <br>
![ultrasounds](assets/ultrasound-montage.png)

### Network architecture
The detection pipeline first passes the images through a convolutional head. It then puts the result through a 19-neuron within an NCP. Below is the wiring diagram of the NCP. Notice the distinction between the different kinds of neurons, as well as where the recurrent connections seem to be:
![wiring-diagram](assets/lnn_auto_wiring_diagram.png)

One of the key advantages of the LNN is its relative sparsity compared to other architectures. This will be apparent in the comparison of the number of parameters later on.

### Training
The LNN model was trained over 50 epochs and a batch-size of 128. Cross entropy-loss was used for the model, as well as the Adam optimizer with a learning rate of 0.001 over. The training:validation:testing ratio was 7:1:2. The validation accuracy was measured after each epoch. 

### Using files
After setting up your environment run ```train.py``` <br>
There are flags within this file to adjust which model to run, whether to save or load a model, etc. <br>
The best LNN model can be found under ```saved_models/LNN_SAVE_898590```

## Results
Below are the figures of the training loss, training accuracy, and validation accuracy across epochs as the model was training.

![training_loss](assets/training_loss.png)
![train_val_accuracy](assets/training_val_acc.png)

When using the final testing set, the LNN model had the following scores: <br>
### **Area under curve (AUC): 0.8897** <br>
### **Accuracy (ACC) : 0.8462**<br>
### **F1-score: 0.8974** <br>


In order to benchmark the results of the LNN, it was compared a traditional deep neural network (DNN) trained on the data and same hyperparameters. The AUC, ACC, and F1-score are compared below.

![auc-comparision](assets/auc.png)
![acc-comparision](assets/acc.png)
![fscore-comparision](assets/F-score.png)

Notice the significant performance increases despite the difference in amount of parameters (120% percent difference): <br>
![param-comparison](assets/param_count.png)

Liquid neural networks are considerably more sparse, which as the original researchers behind LNNs noted has positive implications on interpreting the model and examining the network's attention. And despite the smaller model, **it still manages to outperform the traditional deep neural networks on all metrics** over the breast cancer image classification task.


### Citations
Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid Time-constant Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7657-7666. https://doi.org/10.1609/aaai.v35i9.16936

Lechner, M., Hasani, R., Amini, A. et al. Neural circuit policies enabling auditable autonomy. Nat Mach Intell 2, 642–652 (2020). https://doi.org/10.1038/s42256-020-00237-3

Yang, J., Shi, R., Wei, D. et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Sci Data 10, 41 (2023). https://doi.org/10.1038/s41597-022-01721-8



